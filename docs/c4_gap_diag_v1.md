# C4 差距诊断 v1（c4-gap-diag-v1；2026-11-05；规则在运行前写定，看到结果后不修改）

背景：C4 FaSNet-TAC 的"目标至少命中一个"是 0.395，R1 是 0.575，差 0.18，按 `docs/c4_v1.md` 判为"失真仍是主要障碍"。这里只作诊断，不跑 C5，不改 P2、binding-score-v1、已有结果或判定。

## A. 只读分析：C4 FaSNet-TAC 中"两者都不含"的 129 条
- 按 FaSNet-TAC 目标路的 SI-SDR（direct 参考；来自 `outputs/frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl`）分层：< 5、[5,10)、[10,15)、≥ 15 dB。
- 每层报告：条数，占该 SI-SDR 层全部样本的比例，M（与任一说话人 WER < 0.5），U（和两句都无关），以及转写更接近目标还是干扰者（"明确选择"的定义与 `c3_output_diag.py` 相同）。

## B. C4-direct（GPU）
- 输入：同样 200 个样本中目标说话人的 **direct 参考**（`outputs/frontend_mix_v1_eval_v2/direct_refs/`，8 kHz，已与官方定义核对过）。
- 处理与 C4 完全相同：`resample_poly(x, 2, 1)[:2T]` → 有效帧 RMS 统一到 −30 dBFS（512 点帧，峰值 −40 dB）→ 冻结的 P2，不加方向说明 → 解码设置与 P2 相同。
- 作业配置：<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、48 GB、30 分钟。提交作业前工作区必须干净。

## 比较与解释规则（a = 200 个样本上的"目标至少命中一个"，m = M / 200）
- 比较三个条件：C4 FaSNet-TAC（作业 22014545）、C4-direct、R1（作业 22014155，按每个样本的配对判定）。指标是 binding-score-v1 的四类比例、目标至少命中一个、目标全部命中，以及 M、U。
- 定义 d_sep = a(C4-direct) − a(C4 FaSNet)，d_chain = a(R1) − a(C4-direct)。
- 解释规则：
  - d_sep ≥ 0.10：判为"**分离残留和失真是 C4 差距的主要来源**"；
  - |d_chain| ≥ 0.10：判为"**直达声链路与 R1 链路的差异（无混响的直达声、电平定义）本身有影响**"，并报告正负方向；
  - 两者都 < 0.10：判为"**未能区分**"。
  - 两条可以同时成立。
- 阈值是经验值，不是显著性检验。

## 实现
- `scripts/qwen/c4_neither_by_sisdr.py`（A；只读）；
- `scripts/qwen/qwen_omni_c4_direct.py` 和 `.sbatch`（B）；
- `scripts/qwen/eval_c4_gap.py`（比较）。
- 结果写在 `outputs/qwen_omni_c4_direct/<jobid>/`、`outputs/c4_gap_diag/`。

## 结果（2026-11-05）
- 运行记录：
  - C4-direct：GPU 作业 22015148，<NODE>，14:44:32–14:49:48，退出码 0，MaxRSS 26.6 GB。作业记录的 commit 是 `e3d7d04`，没有未提交的依赖文件；direct_refs 的合并 sha256 记在 `config.json`。CPU dry-run 为 srun 22015015。
  - 评估：`scripts/qwen/eval_c4_gap.py`，在登录节点运行，只读 raw 输出。
- 实现与计划的一处不同：A 部分没有单独写成 `c4_neither_by_sisdr.py`，而是合并在 `eval_c4_gap.py` 里，规则不变。
- 结果在 `outputs/c4_gap_diag/22015148/eval_c4_gap.json`。

### A. C4 FaSNet-TAC 的 129 条"两者都不含"，按目标路 SI-SDR（direct 参考）分层
| SI-SDR 层 | 该层样本数 | 其中两者都不含 | 占该层比例 | M | U | 转写明确选了目标 / 没有明确选择 |
|---|---|---|---|---|---|---|
| < 5 dB | 71 | 49 | 0.69 | 9 | 37 | 9 / 40 |
| [5,10) dB | 114 | 71 | 0.62 | 31 | 40 | 31 / 40 |
| [10,15) dB | 15 | 9 | 0.60 | 5 | 4 | 4 / 5 |
| ≥ 15 dB | 0 | — | — | — | — | — |

- 这 129 条里，没有一条的转写明确选成了干扰者。

### B. C4 FaSNet-TAC、C4-direct、R1 对比（200 个样本）
| 条件 | 只含目标 | 只含干扰 | 两者都不含 | 目标至少命中一个 | 目标全部命中 | 配对翻转准确率 | M | U | 解析失败 / 拒绝执行 |
|---|---|---|---|---|---|---|---|---|---|
| C4 FaSNet-TAC | 0.340 | 0.015 | 0.645 | 0.395 | 0.340 | 0.11 | 112 | 81 | 0 / 0 |
| C4-direct | 0.565 | 0 | 0.435 | **0.670** | 0.565 | 0.36 | 185 | 15 | 0 / 0 |
| R1 | 0.490 | 0 | 0.510 | 0.575 | 0.490 | 0.28 | 176 | 21 | 0 / 0 |

三个条件"两者都含"都是 0。

### 判定（冻结规则）
- d_sep = a(C4-direct) − a(C4 FaSNet) = **+0.275**，≥ 0.10 → **分离残留和失真是 C4 差距的主要来源**。
- d_chain = a(R1) − a(C4-direct) = **−0.095**，绝对值 < 0.10 → 直达声链路与 R1 链路的差异没有达到阈值；C4-direct 比 R1 高 0.095，只作记录。
