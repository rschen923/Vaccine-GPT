# VaccineGPT 1,000 条真实序列训练报告 — 2026-10-07

## 执行范围

输入为已审计的 1,000 条 public-final 蛋白序列，抽样种子 20261007，样本文件
SHA-256：
`919e684b55aedb9215e7b9a5b86aa8383e971715a841715836690e8ccb483793`。
全为真实 VFDB 蛋白序列，但这不是 EIB202 数据，且该文件的 1,000 行标签不具备
已核验的目标菌 E1/E2 来源。

训练运行是**自监督 M1 工程 smoke**，目标是判断真实序列能否经过 CLEF 风格
Encoder A 与 GCE 并完成反向传播：

- 每条序列按稳定 SHA-256 分到 train/validation/test = **700/150/150**；
- 精确相同序列不跨拆分；**未用 MMseqs2，也未排除 90% 同源泄漏**；
- 训练 two-view crop InfoNCE；crop 是对同一序列的 80–100% 重叠片段；
- 表征输入为氨基酸 one-hot，不是 ESM-2；
- 训练到 Encoder A、公有投影与 GCE 自环路径；没有私有组学配对特征和真实图边；
- VFDB membership、IEDB labels、EIB202 H1-H4 都没有作为训练目标；
- H1-H4/M2 未监督训练，因 E1/E2 gold 标签不可用而有意不训练。

## 运行结果

| 项 | 结果 |
|---|---:|
| 样本唯一精确序列 | 1,000 |
| split | train 700 / validation 150 / test 150 |
| epoch 1 train / validation InfoNCE | 1.2194 / 0.4482 |
| epoch 2 train / validation InfoNCE | 0.2860 / 0.1557 |
| epoch 3 train / validation InfoNCE | 0.1631 / 0.1587 |
| epoch 4 train / validation InfoNCE | 0.1771 / 0.0876 |
| epoch 5 train / validation InfoNCE | 0.1219 / 0.1583 |
| 选择的 checkpoint | **epoch 4**，验证损失最低 |
| Test paired-crop retrieval Top-1（训练前/训练后） | 1.00 / 1.00 |
| 训练耗时 | 139.37 s |
| checkpoint SHA-256 | `376dfb49ffb63c101b1377b52b207b49424119338a78455137643775deff1687` |

数据与检查点保存在：

- `..\data\processed\vaccinegpt_real_training_fullm1_261007\training_report.json`
- `..\data\processed\vaccinegpt_real_training_fullm1_261007\m1_real_sample_self_supervised.pt`
- M1→M2 gate smoke: `..\data\processed\vaccinegpt_real_training_fullm1_261007\inference_smoke\trained_m1_sample_chain_report.json`

## 对模型代码的诊断

1. 训练/验证 InfoNCE 明显下降，之后验证损失在 epoch 5 回升；best-checkpoint
   选择已按验证损失选中 epoch 4。此次观察支持保留验证集 checkpoint 选择，
   **不支持用最后 epoch 无条件覆盖最佳模型**。
2. Test paired-crop Top-1 在训练前已为 1.00，因此指标存在天花板，不能作为泛化
   改善证据，也不应继续调模型以追求这个指标。此任务只检查了两份相关 crop
   是否回到同一 instance。
3. 因没有预训练模型、同源簇划分、图边、H1-H4 实测标签和独立生物学终点，
   无法根据这次训练决定模型是否能预测毒力/保护性抗原。**不对目标生物学
   架构作参数调整**；调整 H1-H4 会是没有验证依据的事后追线。
4. 下一步监督训练被数据门阻断：现有 public-final 数据的 E1/E2 gold 行为 0。
   用 VFDB positive-only membership 或 E4 weak labels 当“非毒力/非保护”
   二分类 negative 是错误标签解释，因此 H1-H4 training scripts 必须继续拒绝。

## 核验结论

- **工程训练是否跑通：**是，真实蛋白输入完成 M1 Encoder A + GCE 自监督前向、
  反向、验证、best-checkpoint 写出与哈希核验。
- **训练检查点能否回载并处理完整的 1,000 条真实输入：**是；`z_gctx` 为
  `[1000,128]`，M2 前向处理 1,000 行；由于 M2 未训练，门控将 1,000 行全部保持
  `unranked`，没有输出推荐。
- **是否完成 VaccineGPT 的生物学监督训练：**否。
- **是否证明预设疫苗开发目标：**否，结论仍为 `not_evaluated`。
- **是否需要调整模型代码：**此次只证实了验证损失驱动的 checkpoint 选择必要；
  Top-1 指标无区分度。对生物学决策结构没有足够证据作调整。
