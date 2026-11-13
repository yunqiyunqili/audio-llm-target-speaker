# frontend_mix_v1 评估 v2：加入直达声参考（frontend-mix-v1-eval-v2；2026-11-05；内容和判定在计算前写定，看到结果后不修改）

## 范围
- **只重新评估**现有的 `data/mix_v1/separated_v1/{blstm_mvdr,fasnet_tac}/` 输出（作业 22014183）。不重跑前端，不运行 Qwen，不跑 C4/C5。
- 不修改 mix_v1、已有结果或 v1 的判定（`docs/frontend_mix_v1_plan.md`）。v1 的结果和判定保持原样。
- 不写方向匹配 v2：方向估计方法与 v1 完全相同（`eval_frontend_mix_v1.py: doa`），两路方向相同时仍取第 0 路。

## 直达声参考（与前端训练和 SMS-WSJ 评估的定义相同）
- 定义来自 `../speech_separation/multichannel/data/prepare.py`：
  - hd = `rir_generator.generate(c=343, fs=8000, r=mic0, s=声源位置, L=房间尺寸, reverberation_time=0, nsample=8192, order=0, mtype=omnidirectional)`；
  - y = `extract_piece(fftconvolve(源信号, hd), offset − rir_start(完整 6 通道 RIR), T)` × gain。
- 用在 mix_v1 上：几何取该组 cv_dev93 样例中的 mic 0、该声源所在位置（`source_position[:, pos]`）和房间尺寸；源信号是 SLURP 降到 8 kHz 的信号（`resample_poly` 1/2）；offset 用 mix_v1 的偏移；rir_start 用 mix_v1 manifest 记录的值；**gain 用 mix_v1 manifest 记录的增益**。所以直达声参考和 early、完整 image 使用同样的增益和同样的对齐。
- 环境：rir_generator 0.3.0 只装在原来用的 `../speech_separation/envs/sms_wsj` 里，只读使用，不改动。tfgridnet 环境没有这个包，也不安装。
- 输出：直达声参考写到 `outputs/frontend_mix_v1_eval_v2/direct_refs/`（npz 格式，mic 0），不写 `data/`。

## 先在 cv_dev93 上校验
- 用 `random.Random(2026).sample(sorted(cv_dev93 ids), 50)` 抽 50 组（与 frontend_sanity_v1 的 B、C 相同），按上面的定义重算直达声（官方 gain = 10^(log_weight/20) / std(完整 6 通道 image) / 71），与官方评估使用的 `sms_wsj_direct_v1/cv_dev93/<id>_<k>.wav` 比较。
- **通过条件**：100 个参考的最大绝对差 < 1e-6（官方文件是 float32）。不通过就停下，不做后面的评估。

## 指标
- 两个前端都报告三种参考下的两人平均 SI-SDR、SI-SDRi，以及目标路的 SI-SDR、SI-SDRi；三种参考是 direct、early、完整 image（early 和完整 image 的算法与 v1 相同）。SI-SDR 用原 `tfgridnet/objectives.py: sisdr`。
- 同口径对比：
  - SMS-WSJ test_eval92 的原记录（直达声参考）：BLSTM+MVDR 的 SI-SDRi 为 4.412，FaSNet-TAC 为 14.895；
  - frontend_sanity_v1 在 cv_dev93 50 组上的结果（直达声参考）：4.17 和 15.05。

## 参考匹配
- **改用 direct 参考**做 PIT，目标参考对应的那一路就是目标路。
- 另外报告它与 early 参考匹配（v1）的一致率。

## 方向匹配与 C5 判定（门槛不变）
- 方向匹配的目标路：方法与 v1 相同，取让 |wrap(θ̂_k − θ1)| 最小的那一路，相等时取第 0 路。
- **方向匹配准确率（v2）** = 方向匹配的目标路与 **direct 参考匹配**的目标路一致的比例。
- 只用夹角不在 [0,15) 的 150 个样本计算：
  - 准确率 ≥ 0.85：该前端**可以进入 C5**；
  - 否则**不进入 C5**，并列出失败样本。
- [0,15) 只报告，不参与判定。阈值是经验值。
- 另外报告：两路估计方向完全相同的比例，以及两路方向相差 ≤ 5° 的比例（全部 200 个样本，以及不含 [0,15) 的 150 个样本）。

## FaSNet-TAC 两路的方向
- 用 direct 参考匹配，把两路分别归为 A 路和 B 路，报告每一路估计方向更接近 θ1 的比例，以及 95% 置信区间（以组为单位 bootstrap，10,000 次，`default_rng(2026)`）。
- 按夹角、T60、有效重叠比例、完整 image SIR 分层报告，分层边界与 v1 相同。BLSTM+MVDR 也同样报告一份，作为对照。

## 实现
- 第一步（sms_wsj 环境，CPU，srun）：`scripts/sms_wsj/direct_refs_mix_v1.py`，先做 cv_dev93 校验，再计算 mix_v1 的直达声参考。
- 第二步（tfgridnet 环境，CPU，srun）：`scripts/sms_wsj/eval_frontend_mix_v1_v2.py`。
- 结果：`outputs/frontend_mix_v1_eval_v2/`。

## 结果（2026-11-05；只重新评估已有输出）
- 运行记录：
  - 第一步：srun 22014507（<CPU_PARTITION>，sms_wsj 环境），脚本 commit 为 `2b5264b`。
  - 第二步：srun 22014513（<CPU_PARTITION>，tfgridnet 环境）。
- 结果在 `outputs/frontend_mix_v1_eval_v2/`：`direct_refs/`（npz，约 66 MB，不提交）、`direct_refs_summary.json`、`eval_v2.json`、`eval_v2_per_sample.jsonl`。
- **cv_dev93 校验通过**：50 组共 100 个直达声参考，与官方 `sms_wsj_direct_v1` 的最大绝对差为 1.44e-8（要求 < 1e-6）。
- mix_v1 上用同样方法重算的 early image，与已写出的文件最大差为 2.79e-8。
- 方向估计值与 v1 一致：v2 的脚本对每个样本重算一次 doa 并做断言，结果相同。

### 三种参考下的结果（全部 200 个样本的均值；SI-SDRi = 输出 − 观测 mic 0）
| 前端 | 参考 | 观测 mic 0 的 SI-SDR | 两人平均 SI-SDR | 两人平均 SI-SDRi | 目标路 SI-SDRi |
|---|---|---|---|---|---|
| BLSTM+MVDR | direct | −5.62 | −3.90 | **1.72** | 1.72 |
| BLSTM+MVDR | early | −0.93 | 1.97 | 2.90 | 2.89 |
| BLSTM+MVDR | 完整 | −0.04 | 2.89 | 2.92 | 2.93 |
| FaSNet-TAC | direct | −5.62 | 5.76 | **11.38** | 11.42 |
| FaSNet-TAC | early | −0.93 | −0.45 | 0.47 | 0.36 |
| FaSNet-TAC | 完整 | −0.04 | −1.09 | −1.05 | −1.16 |

### 与 SMS-WSJ 的同口径对比（直达声参考，两人平均 SI-SDRi）
| 前端 | mix_v1 | test_eval92（原记录） | 差距 | cv_dev93 的 50 组（frontend_sanity_v1） | 差距 |
|---|---|---|---|---|---|
| BLSTM+MVDR | 1.72 | 4.41 | −2.69 | 4.17 | −2.45 |
| FaSNet-TAC | 11.38 | 14.90 | −3.51 | 15.05 | −3.67 |

### 参考匹配与方向匹配
| | BLSTM+MVDR | FaSNet-TAC |
|---|---|---|
| direct 参考匹配与 early 参考匹配的一致率（全部 200 个 / 不含 [0,15) 的 150 个） | 0.96 / 0.973 | 0.995 / 1.00 |
| 方向匹配准确率 v2（全部 200 个） | 0.735 | 0.500 |
| **方向匹配准确率 v2（不含 [0,15)，150 个）** | **0.713** | **0.513** |
| **按 v2 判定** | **不进 C5** | **不进 C5** |
| 失败样本数（不含 [0,15)） | 43 | 73 |
| 两路方向完全相同（全部 200 个 / 150 个） | 0.335 / 0.347 | 0 / 0 |
| 两路方向相差 ≤ 5°（全部 200 个 / 150 个） | 0.805 / 0.767 | 0.005 / 0.007 |
| 目标路 / 干扰路方向误差的中位数（全部 200 个） | 2.9° / 2.3° | 77.9° / 2.0° |

- 按夹角分层的方向匹配准确率（[0,15)、[15,30)、[30,60)、[60,180]）：
  - BLSTM+MVDR：0.80 / 0.78 / 0.70 / 0.66；
  - FaSNet-TAC：0.46 / 0.48 / 0.30 / 0.76。
- FaSNet-TAC 各夹角层的 direct SI-SDRi：8.41 / 11.59 / 12.56 / 12.98。
- 按 T60、有效重叠比例、完整 image SIR 的分层结果，以及完整的失败样本清单，都在 `eval_v2.json`。

### FaSNet-TAC 两路估计方向更接近 θ1 的比例（按 direct 参考匹配分为 A 路和 B 路；以组为单位 bootstrap 95% 置信区间）
- 全部 200 个样本：A 路 0.41 [0.34, 0.48]，B 路 0.40 [0.33, 0.47]。BLSTM+MVDR 作为对照：0.475 [0.43, 0.52] 和 0.46 [0.41, 0.51]。
- 按夹角分层（A 路 / B 路）：[0,15) 0.42 / 0.32，[15,30) 0.48 / 0.46，[30,60) 0.30 / 0.34，[60,180] 0.44 / 0.46。
- 按 T60 分层：[0.2,0.3) 0.43 / 0.44，[0.3,0.4) 0.27 / 0.23，[0.4,0.5] 0.47 / 0.44。
- 按有效重叠比例分层：[0,0.5) 0.67 / 0.67（只有 9 个样本），[0.5,0.75) 0.44 / 0.38，[0.75,0.9) 0.35 / 0.35，[0.9,1] 0.43 / 0.50。
- 按完整 image SIR 分层：< −0.5 dB 0.48 / 0.48，[−0.5, 0.5] dB 0.41 / 0.41，> 0.5 dB 0.37 / 0.22。

### 小结（只是描述）
- 用 direct 参考评估后，FaSNet-TAC 在 mix_v1 上的 SI-SDRi 是 11.4 dB，比 SMS-WSJ 同口径低约 3.5 dB；BLSTM+MVDR 是 1.7 dB，比 SMS-WSJ 同口径低约 2.6 dB。
- 两个前端按 v2 都不进 C5：
  - BLSTM+MVDR：两路方向常常相同或很接近（相差 ≤ 5° 的占 77%）；
  - FaSNet-TAC：分离效果好，但目标路的方向估计误差大（中位数 77.9°），而干扰路只有 2.0°。原因还没有查明。
- v1 的结果和判定保持不变。
