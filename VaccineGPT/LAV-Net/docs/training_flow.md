# LAV-Net training flow

Three networks are trained in order; the order cannot be swapped. All three
loops share one schedule and clipping convention: linear warmup over the first
5% of steps, then cosine decay to 1% of the initial rate, gradient-norm cap 1.

```python
import torch
import torch.nn.functional as F

from lavnet import (ATD, ATDConfig, CMCL, CMCLConfig, FrozenPCA, IGP,
                    cosine_warmup, info_nce)

EPOCHS_CMCL, EPOCHS_ATD, EPOCHS_IGP = 200, 300, 200
BS = 128
D_LATENT = CMCLConfig().d


def epoch_batches(n, bs=BS, seed=0):
    """一轮内的批次下标；顺序由种子固定，便于复现。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def fit_pca(x, train_idx, d=D_LATENT):
    """在训练折上拟合主成分并封装为固定投影。

    验证折与测试折只做变换，不参与拟合，否则降维本身即构成泄漏。
    """
    mu = x[train_idx].mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(x[train_idx] - mu, full_matrices=False)
    return FrozenPCA.from_matrix(vh[:d].t())


# 阶段一：跨模态对比（无需标签）
cfg = CMCLConfig(n_cond=fit.shape[1])
cmcl = CMCL(cfg)
# 投影须在训练前换掉随机正交基，否则训练全程都在一个无意义的子空间里做
cmcl.pca_seq = fit_pca(seq, train_idx, cfg.d)
if cfg.use_str:
    cmcl.pca_str = fit_pca(str_, train_idx, cfg.d)

opt = torch.optim.AdamW(cmcl.parameters(), lr=3e-3, weight_decay=1e-2)
sched = cosine_warmup(opt, EPOCHS_CMCL * max(1, -(-len(seq) // BS)))
for epoch in range(EPOCHS_CMCL):
    for idx in epoch_batches(len(seq), BS, seed=epoch):
        out = cmcl(seq[idx], str_[idx], evo[idx], fit[idx])
        loss = info_nce(out["z_a"], out["z_b"], cfg.temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cmcl.parameters(), 1.0)
        opt.step()
        sched.step()

# 冻结编码器，一次性导出联合表征
cmcl.eval()
with torch.no_grad():
    joint = cmcl(seq, str_, evo, fit)["joint"]

# 阶段二：减毒靶点预测（用体内耗竭标签）
atd = ATD(ATDConfig())
opt2 = torch.optim.AdamW(atd.parameters(), lr=1e-3, weight_decay=1e-2)
sched2 = cosine_warmup(opt2, EPOCHS_ATD)
n_pos = y_att.sum().clamp(min=1)
pos_w = (len(y_att) / (2 * n_pos)).reshape(1)
for epoch in range(EPOCHS_ATD):
    out = atd(tokens, joint, ctx, expr)
    loss = F.binary_cross_entropy_with_logits(out["logit"], y_att, pos_weight=pos_w)
    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(atd.parameters(), 1.0)
    opt2.step()
    sched2.step()

# 阶段三：免疫原性守门（正例 + 分层抽样负例）
igp = IGP()
opt3 = torch.optim.AdamW(igp.parameters(), lr=1e-3, weight_decay=1e-1)
sched3 = cosine_warmup(opt3, EPOCHS_IGP)
for epoch in range(EPOCHS_IGP):
    out = igp(joint, ctx)
    loss = F.binary_cross_entropy_with_logits(out["logit"], y_imm)
    opt3.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(igp.parameters(), 1.0)
    opt3.step()
    sched3.step()

# 推理：四门合取。必需性为布尔掩码，不参与概率比较
atd.eval(); igp.eval()
with torch.no_grad():
    p_att = atd(tokens, joint, ctx, expr)["prob"]
    p_imm = igp(joint, ctx)["prob"]
candidate = (p_att > tau_a) & (p_imm < tau_i) & (~ess) & (stab > tau_s)
```

`IGP` trains on `logit`, not `prob`: `binary_cross_entropy_with_logits` does the
log-sum-exp internally, which is numerically stabler than taking sigmoid and
then a logarithm, and it avoids vanishing gradients once probabilities are
clipped to zero. Gating and threshold calibration happen on the `prob` scale;
the two are never mixed.
