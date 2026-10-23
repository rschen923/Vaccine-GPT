"""LAV-Net 冒烟测试与机制验证。

核心验证：对比学习是否真的把 Tn-seq 适应度信息注入序列表征。
做法：合成数据中标签由隐因子决定，该因子同时体现在序列表征与适应度谱上。
若对比对齐成立，则"只用序列表征"的下游预测器在用过 CMCL 之后应当更好——
因为它从适应度谱里间接学到了隐因子。这一检验直接对应真实场景：
新菌株没有 Tn-seq，只能靠序列推断。
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from lavnet import (D, D_CTX, D_EXP, D_EVO, D_SEQ, D_STR, ATD, ATDConfig, CMCL,
                    CMCLConfig, FrozenPCA, IGP, cosine_warmup, count_params,
                    info_nce)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name +
          (("  -- " + str(detail)) if detail else ""))


def make_data(n, seed=0, cfg=None, dim_z=3, world_seed=0):
    """构造检验机制的数据。

    设计要点：标签只由 z2 决定；z2 在适应度谱中很强，在序列表征中极弱。
    于是"仅用序列"的基线应当接近随机，而经对比对齐后应显著改善——
    这正好对应真实场景：新菌株没有 Tn-seq，只能靠序列推断。

    两个种子分工不同。world_seed 决定隐因子到各模态的映射，即"世界"；
    seed 决定抽哪些样本。同一 world_seed 下的训练集与测试集共享同一套
    映射，跨集合评估才有意义。若两者换用不同 world_seed，特征与标签的
    关系在集合间不通用，任何下游指标都会退化到随机，检验也就失去判别力。
    """
    cfg = cfg or CMCLConfig()
    d_seq, d_str, d_evo, n_cond, d = cfg.d_seq, cfg.d_str, cfg.d_evo, cfg.n_cond, cfg.d
    gw = torch.Generator().manual_seed(world_seed)      # 世界参数
    g = torch.Generator().manual_seed(seed)             # 样本
    z = torch.randn(n, dim_z, generator=g)
    z0, z1, z2 = z[:, 0:1], z[:, 1:2], z[:, 2:3]

    # 序列表征：z0/z1 强，z2 弱到接近随机
    b0 = torch.randn(d_seq, generator=gw) * 1.8
    b1 = torch.randn(d_seq, generator=gw) * 1.8
    b2 = torch.randn(d_seq, generator=gw) * 1.8
    seq = (z0 * b0 + z1 * b1 + 0.05 * z2 * b2
           + torch.randn(n, d_seq, generator=g) * 0.8)

    # 结构表征：同序列，噪声更大
    c0 = torch.randn(d_str, generator=gw) * 1.0
    c1 = torch.randn(d_str, generator=gw) * 1.0
    c2 = torch.randn(d_str, generator=gw) * 1.0
    strc = (z0 * c0 + z1 * c1 + 0.05 * z2 * c2
            + torch.randn(n, d_str, generator=g) * 1.0)

    # 适应度谱：z2 强
    f0 = torch.randn(n_cond, generator=gw) * 0.2
    f1 = torch.randn(n_cond, generator=gw) * 0.2
    f2 = torch.randn(n_cond, generator=gw) * 2.0
    fit = (z0 * f0 + z1 * f1 + z2 * f2
           + torch.randn(n, n_cond, generator=g) * 0.3)

    # 残基级 token：只含 z0/z1，刻意不含 z2。
    # 否则 ATD 的残基级路径可以绕过联合表征直接读出标签，
    # 对比对齐的增益就被掩盖，机制检验失去判别力。
    t0 = torch.randn(d, generator=gw) * 1.8
    t1 = torch.randn(d, generator=gw) * 1.8
    L = 40
    tokens = (z0.unsqueeze(1) * t0 + z1.unsqueeze(1) * t1
              + torch.randn(n, L, d, generator=g) * 0.6)

    # 进化信息：z2 中等强度（与 Tn-seq 共同构成模态 B）
    e2 = torch.randn(d_evo, generator=gw) * 1.0
    evo = z2 * e2 + torch.randn(n, d_evo, generator=g) * 0.8

    ctx = torch.randn(n, D_CTX, generator=g) * 0.3
    expr = torch.randn(n, D_EXP, generator=g) * 0.3

    y = (z2.squeeze(1) > 0).float()
    return {"seq": seq, "str": strc, "fit": fit, "evo": evo, "tokens": tokens,
            "ctx": ctx, "expr": expr, "y": y}


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train_cmcl(tr, cfg, epochs=200, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    m = CMCL(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    n, bs = len(tr["y"]), 128
    sched = cosine_warmup(opt, epochs * max(1, -(-n // bs)))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = m(tr["seq"][idx], tr["str"][idx], tr["evo"][idx], tr["fit"][idx])
            loss = info_nce(out["z_a"], out["z_b"], cfg.temperature)
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
            sched.step()
    m.eval()
    with torch.no_grad():
        out = m(tr["seq"], tr["str"], tr["evo"], tr["fit"])
        loss = info_nce(out["z_a"], out["z_b"], cfg.temperature)
    return m, float(loss)


def train_atd(tr, te, cmcl=None, cfg=None, epochs=300, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    cfg = cfg or ATDConfig()
    m = ATD(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    d_seq = cmcl.cfg.d_seq if cmcl is not None else D_SEQ
    pca = FrozenPCA(d_seq, cfg.d, seed=1)      # 基线也降维，保证两条路径维度一致

    def feats(b):
        if cmcl is None:
            return pca(b["seq"])               # 仅原始序列表征，无对比对齐
        with torch.no_grad():
            return cmcl(b["seq"], b["str"], b["evo"], b["fit"])["joint"]

    tr_j, te_j = feats(tr), feats(te)
    n_pos = tr["y"].sum().clamp(min=1)
    w = (len(tr["y"]) / (2 * n_pos)).reshape(1)
    sched = cosine_warmup(opt, epochs)

    for _ in range(epochs):
        out = m(tr["tokens"], tr_j, tr["ctx"], tr["expr"])
        loss = F.binary_cross_entropy_with_logits(out["logit"], tr["y"], pos_weight=w)
        opt.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        sched.step()

    m.eval()
    with torch.no_grad():
        p = m(te["tokens"], te_j, te["ctx"], te["expr"])["prob"].numpy()
    return auc(te["y"].numpy(), p), m


def main():
    print("=== 一、冒烟测试 ===")
    cfg1, cfg2 = CMCLConfig(), ATDConfig()
    m1, m2, m3 = CMCL(cfg1), ATD(cfg2), IGP()
    b = make_data(8, seed=1, cfg=cfg1)

    o1 = m1(b["seq"], b["str"], b["evo"], b["fit"])
    check("CMCL 输出 joint 与两个投影", set(o1) >= {"joint", "z_a", "z_b"}, sorted(o1))
    check("joint 形状 [B,32]", tuple(o1["joint"].shape) == (8, cfg1.d))
    check("投影已归一化",
          torch.allclose(o1["z_a"].norm(dim=-1), torch.ones(8), atol=1e-5))

    o2 = m2(b["tokens"], o1["joint"], b["ctx"], b["expr"], return_tokens=True)
    check("ATD 输出单标量", tuple(o2["logit"].shape) == (8,))
    check("ATD 概率在 0-1", bool(((o2["prob"] >= 0) & (o2["prob"] <= 1)).all()))

    o3 = m3(o1["joint"], b["ctx"])
    check("IGP 输出形状 [B]", tuple(o3["prob"].shape) == (8,))
    check("IGP 的 logit 与 prob 一致",
          torch.allclose(o3["prob"], torch.sigmoid(o3["logit"]), atol=1e-6))

    p1, p2, p3 = count_params(m1), count_params(m2), count_params(m3)
    print(f"     CMCL {p1['_total']:,} | ATD {p2['_total']:,} | IGP {p3['_total']:,}")
    check("CMCL 参数在 9k-11k", 9000 <= p1["_total"] <= 11000, p1["_total"])
    check("ATD 参数在 3k-6k", 3000 <= p2["_total"] <= 6000, p2["_total"])
    check("IGP 参数在 500-900", 500 <= p3["_total"] <= 900, p3["_total"])

    z = F.normalize(torch.randn(16, 32), dim=-1)
    check("InfoNCE 自对齐损失近 0", info_nce(z, z.clone()) < 0.05)
    check("InfoNCE 随机对损失高",
          info_nce(z, F.normalize(torch.randn(16, 32), dim=-1)) > 2.0)

    # 开关
    c_no = ATDConfig(use_cnn=False, use_attention=False, use_adapter=False)
    check("关闭全部基元仍可前向",
          "logit" in ATD(c_no)(b["tokens"], o1["joint"], b["ctx"], b["expr"]))
    check("无残基 token 亦可前向",
          tuple(m2(None, o1["joint"], b["ctx"], b["expr"])["logit"].shape) == (8,))
    m1s = CMCL(CMCLConfig(use_str=False))
    check("关闭结构轨可前向",
          tuple(m1s(b["seq"], None, b["evo"], b["fit"])["joint"].shape) == (8, cfg1.d))

    # 固定投影可载入外部主成分
    mat = torch.linalg.qr(torch.randn(cfg1.d_seq, cfg1.d))[0]
    fp = FrozenPCA.from_matrix(mat)
    check("固定投影可载入外部主成分",
          torch.allclose(fp.weight, mat.float()))

    print()
    print("=== 二、训练正确性（不比性能，只验流程） ===")
    tr = make_data(400, seed=10, cfg=cfg1)
    te = make_data(150, seed=99, cfg=cfg1)

    # 1) 对比损失是否下降
    torch.manual_seed(7)
    m = CMCL(cfg1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
    n, bs = len(tr["y"]), 128
    sched = cosine_warmup(opt, 120 * -(-n // bs))
    g = torch.Generator().manual_seed(7)
    first, last = None, None
    for ep in range(120):
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = m(tr["seq"][idx], tr["str"][idx], tr["evo"][idx], tr["fit"][idx])
            loss = info_nce(out["z_a"], out["z_b"], cfg1.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
        if ep == 0:
            first = tot / cnt
        last = tot / cnt
    print(f"     InfoNCE 首轮 {first:.3f} → 末轮 {last:.3f}")
    check("对比损失下降", last < first, f"{first:.3f}→{last:.3f}")
    check("对比损失未发散", last == last and last < 10, last)

    # 2) 梯度是否有效
    m.zero_grad()
    out = m(tr["seq"][:32], tr["str"][:32], tr["evo"][:32], tr["fit"][:32])
    info_nce(out["z_a"], out["z_b"], cfg1.temperature).backward()
    gsum = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    check("CMCL 梯度非零", gsum > 0, f"{gsum:.2e}")

    # 3) ATD 训练是否可跑通且损失下降
    m2 = ATD(cfg2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-3, weight_decay=1e-2)
    with torch.no_grad():
        j_tr = m(tr["seq"], tr["str"], tr["evo"], tr["fit"])["joint"]
    y = tr["y"]
    sched2 = cosine_warmup(opt2, 80)
    l0 = l1 = None
    for ep in range(80):
        o = m2(tr["tokens"], j_tr, tr["ctx"], tr["expr"])
        loss = F.binary_cross_entropy_with_logits(o["logit"], y)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        sched2.step()
        if ep == 0:
            l0 = loss.item()
        l1 = loss.item()
    print(f"     ATD BCE 首轮 {l0:.3f} → 末轮 {l1:.3f}")
    check("ATD 损失下降", l1 < l0, f"{l0:.3f}→{l1:.3f}")

    # 4) 推理确定性
    m2.eval()
    with torch.no_grad():
        p1 = m2(te["tokens"], m(te["seq"], te["str"], te["evo"], te["fit"])["joint"],
                te["ctx"], te["expr"])["prob"]
        p2 = m2(te["tokens"], m(te["seq"], te["str"], te["evo"], te["fit"])["joint"],
                te["ctx"], te["expr"])["prob"]
    check("推理可复现", torch.allclose(p1, p2))

    # 5) 对比对齐是否带来下游增益
    auc_after, _ = train_atd(tr, te, cmcl=m, cfg=cfg2, epochs=200, seed=3)
    auc_before, _ = train_atd(tr, te, cmcl=None, cfg=cfg2, epochs=200, seed=3)
    print(f"     ATD 测试折 AUC：对比对齐后 {auc_after:.3f}，仅原始序列 {auc_before:.3f}")
    check("对比对齐提升下游判别", auc_after > auc_before + 0.03,
          f"{auc_before:.3f}→{auc_after:.3f}")

    print()
    print("=== 三、基元消融（只验可运行，不比性能） ===")
    variants = [
        ("无适配无卷积无注意力", ATDConfig(use_adapter=False, use_cnn=False, use_attention=False)),
        ("仅适配器", ATDConfig(use_adapter=True, use_cnn=False, use_attention=False)),
        ("适配+卷积", ATDConfig(use_adapter=True, use_cnn=True, use_attention=False)),
        ("适配+注意力", ATDConfig(use_adapter=True, use_cnn=False, use_attention=True)),
        ("全量", ATDConfig(use_adapter=True, use_cnn=True, use_attention=True)),
    ]
    with torch.no_grad():
        j_te = m(te["seq"], te["str"], te["evo"], te["fit"])["joint"]
    ok_all = True
    for name, c in variants:
        try:
            mm = ATD(c)
            with torch.no_grad():
                oo = mm(te["tokens"], j_te, te["ctx"], te["expr"])
            n_par = count_params(mm)["_total"]
            print(f"     {name:22s} 可运行  参数 {n_par:,}")
            ok_all &= tuple(oo["logit"].shape) == (150,)
        except Exception as e:
            print(f"     {name:22s} 失败 {type(e).__name__}")
            ok_all = False
    check("五种基元组合均可前向", ok_all)

    print()
    print("=" * 60)
    print(f"LAV-Net 测试：{len(PASS)} 通过，{len(FAIL)} 失败")
    if FAIL:
        print("失败项：", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

