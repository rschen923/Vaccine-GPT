"""减毒活疫苗设计的两阶段网络。

Stage 1  CMCL：跨模态对比学习
    模态 A = 蛋白序列表征（ESM-3 序列轨 + 结构轨，池化后）
    模态 B = 进化信息 + Tn-seq 适应度谱
    目标   = 让同一基因的两个模态在联合空间对齐，从而把"体内适应度"这一
             实验信息编码进序列表征。无需保护标签，属自监督。

Stage 2  ATD：减毒靶点预测
    输入 = Stage 1 的联合表征 + 残基级 token + 基因组上下文 + 表达
    基元 = 适配器（条件注入）+ 卷积（局部模组）+ 注意力（远距耦合）+ MLP（融合）
    输出 = 单标量，该基因可作减毒靶点的概率

Stage 3  IGP：免疫原性守门（独立网络，输出作硬掩码）
    输出 = 该基因产物为主要抗原的概率；高者不进入候选。

三者为串联的独立网络，非同一网络的多头。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

D_SEQ = 1536      # ESM-3 序列轨池化维数
D_STR = 4096      # ESM-3 结构轨池化维数
D = 32            # 隐层宽度
D_ADP = 8         # 适配器瓶颈
D_CTX = 16        # 基因组上下文特征
D_EXP = 4         # 表达特征
D_EVO = 64        # 进化信息（MSA 保守谱池化 32 + 二肽组成降维 32）


@dataclass
class CMCLConfig:
    """维度全部由配置给出，模型内部不再引用模块级常量。

    这样特征管线改动（保守谱宽度、条件数、是否启用结构轨）只改配置，
    不必回改模型代码，也不会因维度不匹配在训练中途才报错。
    """

    d: int = D
    d_adp: int = D_ADP
    d_seq: int = D_SEQ
    d_str: int = D_STR
    d_evo: int = D_EVO
    n_cond: int = 17          # Tn-seq 条件数，须与适应度宽表列数一致
    proj_dim: int = 32
    temperature: float = 0.1
    use_str: bool = True      # 结构轨缺失时置假，编码器自动退化为单轨


@dataclass
class ATDConfig:
    d: int = D
    d_adp: int = D_ADP
    d_ctx: int = D_CTX
    d_exp: int = D_EXP
    d_hidden: int = 16
    dropout: float = 0.2
    use_adapter: bool = True
    use_cnn: bool = True
    use_attention: bool = True


class FrozenPCA(nn.Module):
    """固定投影，不参与梯度。

    默认以随机正交基占位，仅使前向可运行；正式训练须用 from_matrix
    载入训练折上算得的主成分，否则降维不反映数据分布。
    """

    def __init__(self, d_in: int, d_out: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d_in, d_out, generator=g)
        q, _ = torch.linalg.qr(w)
        self.register_buffer("weight", q.contiguous())

    @classmethod
    def from_matrix(cls, mat) -> "FrozenPCA":
        """由外部主成分矩阵构造。mat 形状 [d_in, d_out]，列须正交。"""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.register_buffer("weight", torch.as_tensor(mat).float().contiguous())
        return obj

    def forward(self, x):
        return x @ self.weight


class Adapter(nn.Module):
    """瓶颈适配器：LN → 降维 → GELU → 升维 → 残差。"""

    def __init__(self, d=D, d_adp=D_ADP):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, d_adp)
        self.up = nn.Linear(d_adp, d)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))


class ProjHead(nn.Module):
    def __init__(self, d_in, d_out, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def cosine_warmup(opt, total_steps: int, warmup_frac: float = 0.05,
                  min_ratio: float = 0.01):
    """线性预热接余弦退火。

    对比学习对温度敏感，固定学习率在后期容易振荡；预热则避免首步
    大批量梯度把投影头推到坍缩点。min_ratio 留一档残余学习率，
    使末段仍在微调而非完全冻结。
    """
    warm = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warm:
            return max(min_ratio, step / warm)
        p = (step - warm) / max(1, total_steps - warm)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ============================================================ Stage 1
class CMCL(nn.Module):
    """跨模态对比学习：序列表征 ↔ 进化信息与 Tn-seq 适应度谱。"""

    def __init__(self, cfg: CMCLConfig | None = None):
        super().__init__()
        self.cfg = cfg or CMCLConfig()
        c = self.cfg

        # 模态 A：序列表征
        self.pca_seq = FrozenPCA(c.d_seq, c.d, seed=1)
        self.pca_str = FrozenPCA(c.d_str, c.d, seed=2) if c.use_str else None
        d_in = c.d * (2 if c.use_str else 1)
        self.enc_a = nn.Sequential(
            nn.Linear(d_in, c.d), nn.GELU(), nn.Linear(c.d, c.d))

        # 模态 B：进化信息 + Tn-seq 适应度谱（合并为一路）
        self.enc_b = nn.Sequential(
            nn.Linear(c.d_evo + c.n_cond, c.d), nn.GELU(),
            nn.Linear(c.d, c.d), nn.GELU(), nn.Linear(c.d, c.d))

        self.proj_a = ProjHead(c.d, c.proj_dim)
        self.proj_b = ProjHead(c.d, c.proj_dim)

    def forward(self, seq_pooled, str_pooled, evo, fitness):
        """Encoder A 取 ESM-3（序列轨 + 结构轨）；Encoder B 取进化信息 + Tn-seq。

        str_pooled 可为 None，此时须以 use_str=False 构造。
        """
        parts = [self.pca_seq(seq_pooled)]
        if self.pca_str is not None:
            parts.append(self.pca_str(str_pooled))
        a = self.enc_a(torch.cat(parts, dim=-1))
        b = self.enc_b(torch.cat([evo, fitness], dim=-1))
        return {
            "emb_a": a, "emb_b": b,
            "z_a": self.proj_a(a), "z_b": self.proj_b(b),
            "joint": 0.5 * (a + b),          # 供 Stage 2 使用
        }


def info_nce(z_a, z_b, temperature=0.1):
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("InfoNCE 输入须为同形状的 [batch, dim] 张量")
    if z_a.size(0) < 2:
        raise ValueError("InfoNCE 每批至少需要两个样本")
    logits = z_a @ z_b.t() / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))


# ============================================================ Stage 2
class ATD(nn.Module):
    """减毒靶点预测：联合表征 + 残基级 token + 上下文 + 表达 → 单标量。"""

    def __init__(self, cfg: ATDConfig | None = None):
        super().__init__()
        self.cfg = cfg or ATDConfig()
        c = self.cfg

        self.adp = Adapter(c.d, c.d_adp) if c.use_adapter else nn.Identity()

        # 局部模组：深度可分离卷积
        self.cnn = nn.Sequential(
            nn.Conv1d(c.d, c.d, 3, padding=1, groups=c.d), nn.GELU(),
            nn.Conv1d(c.d, c.d, 1), nn.GELU(),
        )

        # 远距耦合：单头低秩注意力
        self.q = nn.Linear(c.d, c.d_adp)
        self.k = nn.Linear(c.d, c.d_adp)
        self.v = nn.Linear(c.d, c.d_adp)
        self.o = nn.Linear(c.d_adp, c.d)
        self.att_norm = nn.LayerNorm(c.d)

        # 池化与融合
        self.pool_score = nn.Linear(c.d, 1)
        d_fuse = 2 * c.d + c.d_ctx + c.d_exp
        self.fuse = nn.Sequential(
            nn.Linear(d_fuse, c.d_hidden), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.d_hidden, 1),
        )

    def forward(self, tokens, joint, ctx, expr, return_tokens=False):
        """tokens: [B,L,d] 残基级，可为 None；joint: [B,d] 基因级条件向量。

        残基级路径做卷积与注意力，需 L>1 才有意义。tokens 为 None 时
        退化为单 token，卷积与注意力自动跳过，只保留条件注入与池化，
        使无残基级特征的调用方也能走同一条前向。
        """
        if tokens is None:
            h = self.adp(joint).unsqueeze(1)
        else:
            h = self.adp(tokens) + joint.unsqueeze(1)     # 条件注入

        if self.cfg.use_cnn and h.size(1) > 1:
            h = h + self.cnn(h.transpose(1, 2)).transpose(1, 2)
        if self.cfg.use_attention and h.size(1) > 1:
            q = self.q(self.att_norm(h))
            k, v = self.k(h), self.v(h)
            w = (q @ k.transpose(-1, -2) / math.sqrt(self.cfg.d_adp)).softmax(-1)
            h = h + self.o(w @ v)

        z_attn = (self.pool_score(h).softmax(1) * h).sum(1)
        z_mean = h.mean(1)
        feat = torch.cat([z_attn, z_mean, ctx, expr], dim=-1)
        logit = self.fuse(feat).squeeze(-1)
        out = {"logit": logit, "prob": torch.sigmoid(logit)}
        if return_tokens:
            out["tokens"] = h
        return out


# ============================================================ Stage 3
class IGP(nn.Module):
    """免疫原性守门：预测该基因产物是否为主要抗原。输出作硬掩码。

    同时给出 logit 与 prob。阈值标定与先验估计都在概率尺度上做，
    而损失须在 logit 尺度上算，故两者都要暴露。
    """

    def __init__(self, d: int = D, d_ctx: int = D_CTX, d_hidden: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + d_ctx, d_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hidden, 1))

    def forward(self, joint, ctx):
        logit = self.net(torch.cat([joint, ctx], dim=-1)).squeeze(-1)
        return {"logit": logit, "prob": torch.sigmoid(logit)}


def count_params(m: nn.Module) -> dict:
    out = {}
    for n, sub in m.named_children():
        k = sum(p.numel() for p in sub.parameters() if p.requires_grad)
        if k:
            out[n] = k
    out["_total"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return out

