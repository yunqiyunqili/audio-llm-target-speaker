# 分离前端在 mix_v1 上的零样本推理与评估（frontend-mix-v1；2026-11-05；方法和判定规则在运行前写定，看到结果后不修改）

## 背景
- c3_gate_v1 的结果是"停止，证据不足"，**没有通过**。项目负责人根据因素诊断（主要来源是两人重叠，见 `docs/c3_factor_diag_v1.md`）决定进入分离前端。
- 泄漏检查尚未完成，后续继续按完整 image SIR 分层报告。
- 本阶段**不运行 Qwen**；不修改 c3_gate_v1、mix_v1、P2 prompt、binding-score-v1 或已有结果；不计算配对翻转。

## 前端（零样本，checkpoint 只读）
| 前端 | 代码 | checkpoint（只读） |
|---|---|---|
| BLSTM+MVDR | `../speech_separation/multichannel/blstm_mvdr`（原 `train.py` 中的 `Model`、`normalize`） | `exp/sms_blstm_mvdr_v1/best.pt` |
| FaSNet-TAC | `../speech_separation/multichannel/fasnet_tac` | `exp/sms_fasnet_tac_v1/best.pt` |

- 两个模型都在 SMS-WSJ 上训练，训练目标是 mic 0 的直达声。这次是**零样本**使用，不训练，也不微调。
- 加载前核对：checkpoint 的 sha256 与原记录（`test_summary.json`）一致；`package['config']` 与 `run_config.json` 一致。
- 推理方式与原 `evaluate()` 相同：x = 观测 / std(观测)，输出 pred = model(x) × std。输出是两路 mic 0 上的单通道估计，形状 (2, T)。FP32，batch 1，随机种子和确定性设置与原代码相同。
- 环境：`<CONDA_ROOT>/envs/tfgridnet`，不改动。作业配置：<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、32 GB、1 小时。不写 `__pycache__` 到 `../`。

## 输入与输出
- 输入：mix_v1 的 200 个 `data/mix_v1/devel_seed2026/gNNN_sS/obs_6ch_8k.wav`（作业 22013065）。
- 输出：`<SHARED_ROOT>/data/mix_v1/separated_v1/<frontend>/gNNN_sS/out0.wav` 和 `out1.wav`（8 kHz，单通道，float32），另有一个 `manifest.jsonl`。目录已存在就报错，**不覆盖**已有文件。
- 预计磁盘占用约 70 MB（2 个前端 × 200 个样本 × 2 路 × 平均约 2.6 s × 8000 × 4 B）。超过 2 GB 就停下报告。

## 参考信号
- **early image**：mix_v1 已经写出的 `early_{A,B}_6ch_8k.wav`，取 mic 0。
- **完整 image**：mix_v1 没有写出这个文件。用与 `scripts/sms_wsj/mix_slurp_v1.py` 相同的函数和 manifest 中的增益，在内存里重新计算 mic 0 的完整 image，不写文件。核对方法：用同样的方式重算 early image，与已写出的文件比较，最大绝对差必须 < 1e-6（float32 精度），否则停止。
- 两个前端使用完全相同的参考信号。

## 指标
- SI-SDR：与原 `tfgridnet/objectives.py: sisdr` 相同（零均值、尺度不变）。
- 每个样本，两路输出和 [目标, 干扰者] 两个参考做 PIT，取两人平均 SI-SDR 较大的那个排列（与原 `scores()` 相同）。
- 报告：
  - 两人平均 SI-SDR，以及目标一路的 SI-SDR；
  - SI-SDRi = SI-SDR − SI-SDR(观测 mic 0, 参考)；
  - early 和完整 image 两种参考分别计算。
- 分层（只作探索）：夹角 [0,15)、[15,30)、[30,60)、[60,180]；T60 [0.2,0.3)、[0.3,0.4)、[0.4,0.5]；有效帧重叠比例 [0,0.5)、[0.5,0.75)、[0.75,0.9)、[0.9,1]；完整 image SIR < −0.5、[−0.5, 0.5]、> 0.5 dB。重叠比例和 SIR 取自生成作业的 manifest。

## 参考匹配（C4，只作离线诊断）
- 用 early image 参考做 PIT，目标参考所对应的那一路记为"参考匹配的目标路"。另外报告用完整 image 参考时选出的目标路，以及两者是否一致。

## 方向匹配（不使用参考信号，也不使用说话人身份）
只用输出路 y_k、6 麦观测 x_m，以及阵列几何（cv_dev93 样例中的麦克风坐标，`sms_wsj.json`）：
1. STFT：Hann 窗 512 点，hop 128（8 kHz）。只用 80–3900 Hz 的频点。
2. 对每路 k、每个麦克风 m = 1…5：G_{k,m}(f) = Σ_t conj(Y_k(t,f)) · X_m(t,f)，然后做 PHAT 归一化：Ĝ = G / max(|G|, 1e-12)。
3. 远场平面波、只考虑水平面（各点的 z 坐标相近，忽略高度）：对房间方位 φ（0–359°，步长 1°），τ_m(φ) = −((p_m − p_0)·u(φ)) / c，其中 u(φ) = (cos φ, sin φ)，c = 343 m/s，p 为麦克风的 x-y 坐标。
4. 打分 S_k(φ) = Σ_m Re Σ_f Ĝ_{k,m}(f) · exp(j2πf τ_m(φ))；估计方向 φ̂_k = argmax S_k。
5. 换算成相对 mic 0 的方位：θ̂_k = wrap(φ̂_k − az(mic0 − 阵列中心))，与 az_rel 的定义相同（`scripts/sms_wsj/sms_wsj_geometry_stats.py`）。
6. **方向匹配的目标路** = 让 |wrap(θ̂_k − θ1)| 最小的那一路。两路一样近时取 k = 0。
- 方法本身的检查（只作诊断，用到了参考信号，不参与任何匹配或判定）：把参考 early image 当作 y_k 代入同样的流程，报告估计出的方向与 θ1、θ2 的误差。

## 判定（每个前端分别判断；阈值是经验值）
- **方向匹配准确率** = 方向匹配的目标路与参考匹配（early image）的目标路一致的比例。
- 只用夹角**不在 [0,15)** 的样本（150 个）计算：
  - 准确率 ≥ 0.85：该前端**可以进入 C5**；
  - 准确率 < 0.85：该前端**暂不进入 C5，只做 C4**，并列出失败样本。
- [0,15) 那一层只报告，不参与判定。
- 另外报告：FaSNet-TAC 和 BLSTM+MVDR 各自的单路方向估计误差（参考匹配的目标路的 θ̂ 与 θ1 的差，干扰路的 θ̂ 与 θ2 的差），并按上面的分层列出准确率。
- 同一组的两个互换样本都要保留并分别计算，不计算配对翻转。

## 实现
- 推理：`scripts/sms_wsj/frontend_mix_v1.py` 和 `.sbatch`（tfgridnet 环境）。
- 评估：`scripts/sms_wsj/eval_frontend_mix_v1.py`（tfgridnet 环境，CPU，用 srun 运行）。

## 结果（作业 22014183；**待流程核查，暂不作前端性能结论**）
> 以下数字是按本计划实际运行得到的，但推理流程、混合流程和时长影响还没有核查（见 `docs/frontend_sanity_v1.md`）。核查完成前，**不据此对前端性能下结论**。

- 推理：作业 22014183，<NODE>，2026-11-05 12:44:22–12:46:10，退出码 0；commit `944de58`，运行时没有未提交的文件；CPU dry-run 为 srun 22014178。两个 checkpoint 的 sha256 和配置都与原记录一致（BLSTM+MVDR 第 62 轮，FaSNet-TAC 第 98 轮）。
- 输出：`data/mix_v1/separated_v1/{blstm_mvdr,fasnet_tac}/`，各 33 MB。
- 评估：`scripts/sms_wsj/eval_frontend_mix_v1.py`（commit `bd2d1b0`），srun 22014185，结果在 `outputs/frontend_mix_v1_eval/22014183/`。
- 完整 image 参考是在内存里重算的。核对：用同样方法重算的 early image 与已写出的文件最大差 2.8e-8。
- 方向估计器的检查：把参考 early image 代入，误差中位数为目标 0.5°、干扰 0.6°，全部样本误差 ≤ 10°。

| | 参考 | 两人平均 SI-SDR | 两人平均 SI-SDRi | 目标路 SI-SDR | 目标路 SI-SDRi |
|---|---|---|---|---|---|
| 混合信号 mic 0（基线） | early / 完整 | −0.93 / −0.04 | — | — | — |
| BLSTM+MVDR | early | 1.97 | 2.90 | 1.98 | 2.89 |
| BLSTM+MVDR | 完整 | 2.89 | 2.92 | 2.90 | 2.93 |
| FaSNet-TAC | early | −0.45 | 0.47 | −0.56 | 0.36 |
| FaSNet-TAC | 完整 | −1.09 | −1.05 | −1.19 | −1.16 |

| | BLSTM+MVDR | FaSNet-TAC |
|---|---|---|
| early 参考和完整参考选出的目标路一致 | 0.99 | 1.00 |
| 方向匹配准确率（全部 200 个样本） | 0.725 | 0.495 |
| 方向匹配准确率（不含 [0,15)，150 个样本） | 0.713 | 0.513 |
| 按 v1 规则判定 | 不进 C5，只做 C4 | 不进 C5，只做 C4 |
| 失败样本数（不含 [0,15)） | 43 | 73 |
| 目标路方向误差中位数 / 干扰路方向误差中位数 | 3.1° / 2.3° | 77.9° / 1.9° |
| 有一路接近静音的样本数 | 0 | 4 |

- 按夹角分层的方向匹配准确率：BLSTM+MVDR 依次为 0.76 / 0.82 / 0.68 / 0.64；FaSNet-TAC 依次为 0.44 / 0.48 / 0.30 / 0.76（[0,15)、[15,30)、[30,60)、[60,180]）。
- BLSTM+MVDR 不含 [0,15) 的 150 个样本中：
  - 52 个样本两路的估计方向完全相同，按规则取第 0 路，这部分准确率 0.48；
  - 115 个样本两路方向相差 ≤ 5°；
  - 两路方向相差 > 5° 的 35 个样本，准确率 0.97。
- v1 的判定保持不变，不作修改。
