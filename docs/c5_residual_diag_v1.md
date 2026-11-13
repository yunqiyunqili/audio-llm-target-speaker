# C5 残差拆分 v1（c5-residual-diag-v1；2026-11-05；只读，在计算前写定；只作描述，**不写成因果**）

- 对象：C5 FaSNet-TAC 的 200 个目标路。按 C5 的极性修正后使用，8 kHz，与 separation-quality-diag-v1 相同。
- 参考（mic 0，用 mix_v1 的增益和对齐）：
  - direct：`outputs/frontend_mix_v1_eval_v2/direct_refs/`；
  - early：mix_v1 已写出的 `early_{A,B}_6ch_8k.wav`；
  - 完整 image：在内存里重算，方法与 eval v2 相同。
- 每个声源拆成三部分：直达声 d，早期反射 r_e = early − d，晚期混响 r_l = 完整 − early。
- 嵌套最小二乘（所有信号先做零均值）：
  1. 基 B1 = {d_T, d_I}，残差能量 R1；
  2. 基 B2 = B1 ∪ {r_e,T, r_e,I}，残差能量 R2；
  3. 基 B3 = B2 ∪ {r_l,T, r_l,I}，残差能量 R3。
- 描述量：
  - 早期反射能解释的比例 = (R1 − R2) / R1；
  - 晚期混响能解释的比例 = (R2 − R3) / R1；
  - 仍然无法解释的比例 = R3 / R1。
  - 另外报告只用早期反射或只用晚期混响时，与 direct 相比各自额外解释的比例：用基 {d, r_e} 和基 {d, r_l} 分别计算，作为参考。
- 注意：
  - 早期反射、晚期混响和直达声之间相关性很强，最小二乘的分配不唯一，**这些比例只用来描述"残差中有多少可以用这些成分线性表示"**；
  - 不判断残差是由什么造成的。
- 报告全部 200 个样本的中位数和四分位数，并按 C5 的"目标至少命中一个"分成两组分别报告。
- 试听样本：用 `random.Random(2026).sample(sorted(200 个 key), 10)` 抽 10 个，列出：
  - C5 目标路的文件路径（`data/mix_v1/separated_v1/fasnet_tac/gNNN_sS/out{k}.wav`，原始极性）；
  - 目标的 direct 参考。它目前是 npz 格式，为了试听，把这 10 个导出成 wav（8 kHz，float32），放在 `outputs/c5_residual_diag_v1/listen/`，不写 `data/`。
- 实现：`scripts/sms_wsj/c5_residual_diag.py`（tfgridnet 环境，CPU，srun）。

## 结果（2026-11-05；描述性的最小二乘分解，各成分相关性强，**不是因果**）
- 运行记录：srun 22015449（<CPU_PARTITION>，tfgridnet 环境），脚本 commit 为 `4caeb4b`。结果在 `outputs/c5_residual_diag_v1/{residual.json,per_sample.jsonl}`；试听用的 wav 在 `listen/`。

| 占"direct 投影后残差能量"的比例 | 全部 200 个（P25 / 中位数 / P75） | C5 命中（79 个）中位数 | C5 未命中（121 个）中位数 |
|---|---|---|---|
| 可由早期反射线性解释 | 0.018 / **0.047** / 0.086 | 0.045 | 0.049 |
| 再由晚期混响线性解释 | 0.0014 / **0.0043** / 0.0095 | 0.0043 | 0.0036 |
| 仍无法解释 | 0.905 / **0.945** / 0.972 | 0.948 | 0.941 |

- 只用晚期混响（{d, r_l}）时，额外解释的比例中位数是 0.0040。只用早期反射时与嵌套分解相同，中位数 0.047。
- 描述：C5 目标路的残差里，约 95%（中位数）不能用目标或干扰者的直达声、早期反射、晚期混响线性表示。命中和未命中两组之间这几个比例差别很小。
- 试听样本（`random.Random(2026).sample`，共 10 个；C5 目标路文件在 `data/mix_v1/separated_v1/fasnet_tac/<key>/out{k}.wav`，是原始极性；direct 参考在 `outputs/c5_residual_diag_v1/listen/<key>_direct_target_8k.wav`）：

| key | C5 目标路 | C5 是否命中 | 目标句 |
|---|---|---|---|
| g015_s1 | out1 | 否 | where is george w bush from |
| g040_s2 | out0 | 否 | i like to listen to jazz |
| g064_s1 | out0 | 否 | how tall is micheal jackson |
| g065_s2 | out0 | 是 | what do i have scheduled for next week |
| g082_s2 | out1 | 是 | what is a basketball |
| g013_s1 | out1 | 否 | show results for cooking lasagna from web |
| g028_s2 | out0 | 是 | can you tell me what haggis looks like |
| g076_s2 | out1 | 否 | start playing jazz music |
| g079_s2 | out0 | 否 | what are my reminders today |
| g071_s1 | out1 | 否 | today's weather is awesome |
