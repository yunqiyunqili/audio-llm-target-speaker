# 更强前端对比 v1（frontend-compare-v1；**草案**，2026-11-05；未提交任何作业）

## Checkpoint 盘点（2026-11-05 约 21:10，只读；正在写入的运行目录只看了文件元数据，没有读取内容）
根目录 `MC = <SEPARATION_REPO>/multichannel`。

| 运行 | 状态 | 依据 | 是否可用 |
|---|---|---|---|
| TF-GridNet 6ch `MC/tfgridnet/exp/sms_direct_6ch_v1` | **训练中**：作业 22009884（sms-grid-6ch）RUNNING，已运行约 21 小时 | best.pt / latest.pt / metrics.jsonl 最后修改于 2026-11-05 15:45:53；**没有 test_summary.json** | 否：没有完成，没有明确的 test 指标 |
| TF-GridNet 1ch `MC/tfgridnet/exp/sms_direct_1ch_v1` | 训练中：作业 22010327 RUNNING | latest.pt 最后修改于 18:44:41；没有 test_summary.json | 否；而且是单通道 |
| SpatialNet-small `MC/spatialnet/exp/sms_spatialnet_small_v1` | **训练中**：作业 22010857 RUNNING，已运行约 13 小时 | best.pt 最后修改于 20:34:09；没有 test_summary.json | 否 |
| TF-GridNet pilot_6ch / smoke_6ch，SpatialNet smoke | 已完成，但都是 smoke：1 个 epoch，只测 5 条 | test_summary：SI-SDRi −12.99 / −12.81 / −6.91 dB；checkpoint sha256 分别以 `ac9fdef2…`、`b2d36e69…`、`d3ead726…` 开头（取自 test_summary） | 否：只是流程检查 |
| `repos/TFGridNet_8k_random` | 训练中：作业 21972919（tfgridnet-8k-random）RUNNING | 单通道，另一套数据（不是 SMS-WSJ 6 麦） | 否 |

- 当前 epoch 数、验证集指标：这些信息在正在写入的 `metrics.jsonl` 里，按要求**没有读取**。2026-11-03 的记录（`docs/checkpoints.md`）是：TF-GridNet 6ch 当时 9/100 个 epoch，dev SI-SDR 16.42 dB；SpatialNet 43/100 个 epoch，best dev 19.30 dB（epoch 42）。这两个数字现在已经过时。
- 输入：两个模型都是 6 通道、8 kHz，目标是 mic 0 的直达声，输出 (B, 2, T)，与 FaSNet-TAC 的协议相同。
- **结论：现在没有"已经完成、指标明确"的 TF-GridNet 或 SpatialNet checkpoint**，本草案暂时不能运行。

## 进入运行的前提（全部满足才运行）
1. 运行目录里有 `test_summary.json`，checkpoint 中 `complete == True`，并且产生它的训练作业已经结束，不在运行或排队。
2. `best.pt` 的 sha256 与 `test_summary.json` 记录的一致，package config 与 `run_config.json` 一致。
3. 只读取已经完成的文件，不碰正在运行的训练作业。

## 方法（与 FaSNet-TAC 的 C5 链路相同，零样本）
- 输入：mix_v1 的 200 个 `obs_6ch_8k.wav`。推理方式与 `frontend_mix_v1.py` 相同：std 归一化 → model → × scale。输出写到 `data/mix_v1/separated_v1/<frontend>/`，不覆盖已有文件。
- 选路：与 v3 相同。零延迟相关做极性检查 → `doa` → 取最接近询问方位 θ1 的一路，一样近时取第 0 路。
  - **依赖 oracle 方位**，与 C5 相同。
  - 选路只用输出、观测和阵列几何，不用参考信号。
- 分离指标：direct 参考下的两人平均 SI-SDRi 和目标路 SI-SDRi；参考是 `outputs/frontend_mix_v1_eval_v2/direct_refs`。
- 选路准确率：以 direct 参考匹配为标准，报告全部样本和不含 [0,15) 的两种口径。
- 下游，对选出的目标路分别评估：
  - **P2**：8→16 kHz → −30 dBFS → 冻结的 P2，与 C5 的链路相同；
  - **Whisper + T2**：同样的音频 → whisper-large-v3（`06f233fe…`）→ 冻结的 T2，与 C7W-C5 的链路相同。
- 指标：`eval_c4_v1.score` 和 `summ`，包括四类比例、目标至少命中一个、目标全部命中、配对翻转准确率、M、U，以及 Whisper 的 WER。

## 判定（经验值，不是显著性检验；不使用任何 test 数据）
- 两条链路分别与 FaSNet-TAC 比较（a = 200 个样本上的"目标至少命中一个"）：
  - P2：a(新前端 + P2) ≥ a(FaSNet-TAC C5) + 0.10，即 ≥ 0.395 + 0.10 = **0.495**；
  - Whisper + T2：a(新前端 + Whisper + T2) ≥ a(FaSNet-TAC C7W-C5) + 0.10，即 ≥ 0.340 + 0.10 = **0.440**。
- 满足哪一条，就对那一条链路判为"**更强前端有明显帮助**"；不满足则判为"不能认定有明显帮助"。两条链路分别报告。
- SMS-WSJ 的 test 指标只用来确认 checkpoint 是否合格（来自已有的 test_summary.json），不计算新的 test 结果，也不使用 SLURP test。

## 预计资源（估计，以实测为准）
- 前端推理（tfgridnet 环境）：每个前端一个作业，<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、32 GB、1 小时。200 个短样本（约 1.5–4.6 s）估计几分钟；输出约 33 MB。
- 下游（qwen_omni 环境）：每个前端一个作业，<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、48 GB、30 分钟。
  - 估计：P2 200 条约 4–7 分钟（音频输入实测每条 1.1–1.9 s），Whisper 约 1 分钟，T2 约 3.5 分钟，加载约 1–1.5 分钟，合计约 10–14 分钟。
  - 22016221 那次超时说明这个估计可能偏低。必要时可以把 P2 和 Whisper + T2 拆成两个作业。
- 不需要下载任何东西：Whisper 已经有了，Qwen 也已经有了。

## 训练进度（2026-11-05 21:12，只读了 continuation_jobs.txt 和 metrics.jsonl；没有读 checkpoint 内容，没有计算 sha256）
| | TF-GridNet 6ch（`sms_direct_6ch_v1`） | SpatialNet-small（`sms_spatialnet_small_v1`） |
|---|---|---|
| 续训作业链（continuation_jobs.txt） | 21986967 → 21989292 → 22009884（RUNNING） | 21988892 → 21995307 → 22010857（RUNNING） |
| 当前已完成的 epoch | **16** / 100 | **72** / 100 |
| 最佳验证 dev SI-SDR | **17.565 dB，第 15 个 epoch**（第 16 个 epoch 是 17.537 dB；dev loss 最小的是第 16 个 epoch） | **19.943 dB，第 72 个 epoch**（dev loss 最小的也是第 72 个 epoch） |
| 最近 5 个 epoch 的 dev SI-SDR | 12:17.11，13:17.26，14:17.19，15:17.57，16:17.54 | 68:19.77，69:19.65，70:19.82，71:19.94，72:19.94 |
| 最近 5 个 epoch 平均训练用时 | 27,423 s（约 7.6 小时） | 6,360 s（约 1.8 小时） |
| metrics.jsonl 最近更新时间 | 2026-11-05 15:45:53（3,921 B） | 2026-11-05 20:34:09（18,784 B） |
| best.pt | 66,211,644 B，2026-11-05 15:45:53 | 14,618,898 B，2026-11-05 20:34:09 |
| latest.pt | 66,236,552 B，2026-11-05 15:45:53 | 14,670,992 B，2026-11-05 20:34:08 |
| 下一个 epoch 的学习率 | 0.001 | 4.85e-4 |

- TF-GridNet 的 best.pt 和第 16 个 epoch 的 metrics 是同一时刻写入的，所以 best.pt 很可能对应第 16 个 epoch，也就是按 dev loss 选的，而不是 dev SI-SDR 最好的第 15 个 epoch。这只是从修改时间推断出来的，没有读 checkpoint 内容确认。
- 两个模型都**没有完成训练，也没有 test 指标**。

## 诊断快照规则（草案，2026-11-05；运行前需要另行批准）
用于在训练完成之前，先拿正在训练的 checkpoint 做诊断。**快照的结果只算诊断，不算完成的模型，也不使用任何 test 数据。**
1. **选择**：选验证指标（dev SI-SDR）最好、并且**已经写完**的 checkpoint。如果 best.pt 对应的 epoch 不是 dev SI-SDR 最好的那个，就如实记录两者的差别；最终用哪个文件，按"已写完的文件里 dev SI-SDR 最高"来定，并在运行前写明。
2. **已写完的判断**：复制前记录文件大小和修改时间，至少隔 60 秒再记一次，两次一致，并且这段时间 metrics.jsonl 也没有更新，才算已写完。
3. **复制**：只读复制到共享盘新目录 `<SHARED_ROOT>/checkpoints/snapshots/<model>_ep<epoch>_<时间>/`，**不往训练目录写任何东西**。复制后再记一次源文件的大小和修改时间，必须与复制前一致，否则作废重来；目标文件的大小也必须与源文件一致。
4. **记录**：原路径、epoch、dev SI-SDR、文件大小、修改时间，以及副本的 sha256（只对副本计算）。写入一个 `snapshot.json` 和本文件。
5. **加载检查**：从副本加载，package config 必须与 `run_config.json` 一致，记录 `complete` 字段的值（预计为 False）。
6. **先在 cv_dev93 上核查**：用与 frontend_sanity 相同的 50 组（`random.Random(2026).sample`），在内存中推理，用官方 direct 参考计算 SI-SDRi，并检查两路输出的极性，规则与 v3 相同。这一步只是确认流程可以用，不设通过门槛。
7. 然后按本文件前面的"方法"运行，**判定门槛不变**：P2 ≥ 0.495；Whisper + T2 ≥ 0.440。结果标注为"诊断快照"。

## 诊断快照：已批准的执行方式（2026-11-05；写于复制之前）
- **TF-GridNet 6ch 和 SpatialNet-small 都使用复制时的 `best.pt`**。实际的 epoch 和验证指标从**副本**里读取并记录（package 中的 `epoch`、`best`、`history`），不从训练目录读取 checkpoint 内容。
- 快照目录：`<SHARED_ROOT>/checkpoints/snapshots/<model>_ep<epoch>_<YYYYmmdd-HHMMSS>/`，里面有 `best.pt` 和 `snapshot.json`。
- **这是训练中的快照，只作诊断，不代表完成的模型**；不使用 test。
- **cv_dev93 的停止条件**（对每个模型分别判断）：推理报错、输出形状不是 (2, T)、或者 50 组上的平均 direct SI-SDRi ≤ 0（即不比混合信号好），就停止，不进入 mix_v1。
- 通过后：
  1. mix_v1 的 200 个样本做零样本推理，输出到 `data/mix_v1/separated_v1_snapshot/<model>_ep<epoch>/`；
  2. 按 v3 流程评估 direct SI-SDRi、极性和选路准确率；
  3. 用与 C5 相同的选路选出目标路，分别用 P2 和 Whisper + T2 评估。
- 判定门槛不变：P2 ≥ 0.495；Whisper + T2 ≥ 0.440。两个模型分别报告。
- 每个作业提交前工作区必须干净；记录 commit、节点、GPU 型号和每条耗时。

## 诊断快照结果（2026-11-05；**快照诊断，不代表完成的模型**；没有使用 test）
### 快照（srun 22016808；规则见上；源文件的大小和修改时间在相隔 65 秒的两次检查和复制后都没有变化；训练目录没有任何写入）
| 模型 | 原路径 | 副本路径 | epoch | 验证 dev SI-SDR | best（选择标准） | complete | 副本 sha256 |
|---|---|---|---|---|---|---|---|
| TF-GridNet 6ch | `MC/tfgridnet/exp/sms_direct_6ch_v1/best.pt`（66,211,644 B，2026-11-05 15:45:53） | `checkpoints/snapshots/tfgridnet_ep16_20261105-211918/best.pt` | **16** | 17.537 dB | 0.70188（dev loss，越小越好） | False | `660fe57fbc31e0ed8cae0cd29e9960bc756883f37d45ec05ab626baa54451cf8` |
| SpatialNet-small | `MC/spatialnet/exp/sms_spatialnet_small_v1/best.pt`（14,618,898 B，2026-11-05 20:34:09） | `checkpoints/snapshots/spatialnet_ep72_20261105-211918/best.pt` | **72** | 19.943 dB | −19.9425（dev loss） | False | `582b2a3a10275866a4d379c8c43501e6482e41c39e6d6f1131f40f8444309a3c` |

- TF-GridNet 的 best.pt 是按 dev loss 选的，对应第 16 个 epoch，不是 dev SI-SDR 最好的第 15 个 epoch（17.565 dB）。
- SpatialNet 的 config 与 run_config.json 只有 4 个 model 字段的 tuple 和 list 序列化差异，经过 JSON 往返转换后完全相等（srun 22016812）。

### 前端（作业 22016833：<NODE>，A100-PCIE-40GB，21:37:55–21:43:40，退出码 0；commit `bfe2d24`，DIRTY 为 0；CPU dry-run 为 srun 22016815）
| | TF-GridNet 快照 | SpatialNet 快照 | FaSNet-TAC（对照，完成的模型） |
|---|---|---|---|
| cv_dev93 50 组：direct SI-SDRi（混合信号 SI-SDR 为 −5.61 dB） | **23.16 dB** → 通过 | **25.57 dB** → 通过 | 15.05 dB（frontend_sanity_v1） |
| cv_dev93：两路与 direct 参考相关为负的比例（第 0 路 / 第 1 路） | 0 / 0 | **1.00 / 1.00** | 1.00 / 0 |
| cv_dev93：选路准确率，极性修正前 → 修正后 | 1.00 → 1.00 | 0.31 → **1.00** | 0.64 → 1.00 |
| cv_dev93：每条推理用时 | 1.29 s | 0.11 s | — |
| mix_v1 200 个样本：direct SI-SDRi（两人平均 / 目标路） | **17.66 / 17.78 dB** | **18.56 / 18.59 dB** | 11.38 / 11.42 dB |
| mix_v1：选路准确率（全部 / 不含 [0,15)） | 0.99 / **1.00** | 1.00 / **1.00** | 0.99 / 1.00 |
| mix_v1：被取反的比例（第 0 路 / 第 1 路） | 0 / 0 | 1.00 / 1.00 | 1.00 / 0 |

- mix_v1 输出：`data/mix_v1/separated_v1_snapshot/{tfgridnet_ep16_…,spatialnet_ep72_…}/`，各 33 MB。评估：srun 22016898，结果在 `outputs/frontend_snapshot_eval/`。
- 按夹角分层的 direct SI-SDRi（[0,15)、[15,30)、[30,60)、[60,180]）：TF-GridNet 16.66 / 17.73 / 18.13 / 18.11 dB；SpatialNet 17.45 / 18.95 / 18.85 / 18.97 dB。

### 下游（选路与 C5 相同，依赖 oracle 方位；作业 22016929 TF-GridNet，<NODE>；22016930 SpatialNet，<NODE>；都是 A100-PCIE-40GB；commit `4eb0bba`，没有未提交的依赖文件；CPU dry-run 为 srun 22016911）
| 条件 | 只含目标 | 只含干扰 | 两者都不含 | 目标至少命中一个 | 目标全部命中 | 配对翻转准确率 | M | U | 解析失败 / 拒绝执行 |
|---|---|---|---|---|---|---|---|---|---|
| TF-GridNet 快照 + P2 | 0.510 | 0.010 | 0.480 | **0.595** | 0.510 | 0.29 | 166 | 30 | 1 / 0 |
| SpatialNet 快照 + P2 | 0.530 | 0 | 0.470 | **0.635** | 0.530 | 0.33 | 170 | 29 | 0 / 0 |
| FaSNet-TAC C5（P2，对照） | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 | 82 | 0 / 0 |
| TF-GridNet 快照 + Whisper + T2 | 0.480 | 0.005 | 0.515 | **0.545** | 0.480 | 0.31 | 160 | 36 | 0 / 0 |
| SpatialNet 快照 + Whisper + T2 | 0.510 | 0.005 | 0.485 | **0.590** | 0.510 | 0.32 | 171 | 27 | 1 / 1 |
| FaSNet-TAC C7W-C5（对照） | 0.290 | 0.005 | 0.705 | 0.340 | 0.290 | 0.13 | 109 | 87 | 3 / 3 |
| C4-direct（P2，上限参考） | 0.565 | 0 | 0.435 | 0.670 | 0.565 | 0.36 | 185 | 15 | 0 / 0 |

- Whisper 与目标句的 WER：TF-GridNet 0.198（WER < 0.5 的有 159 条），SpatialNet 0.157（173 条）。对照：FaSNet-TAC C5 0.452（109 条），direct 参考 0.078。
- 每条推理用时（平均 / 最大，单位 s）：TF-GridNet 作业 P2 1.168 / 7.19，Whisper 0.198 / 0.93，T2 1.082 / 1.78；SpatialNet 作业 P2 1.154 / 4.74，Whisper 0.204 / 0.69，T2 1.097 / 1.77。
- 评估：`scripts/qwen/eval_snapshot_downstream.py`，结果在 `outputs/frontend_snapshot_downstream_eval/eval.json`。

### 判定（门槛不变；**快照诊断，不代表完成的模型**）
| 模型 | P2：a（门槛 0.495；比 FaSNet 高多少） | 判定 | Whisper + T2：a（门槛 0.440；比 FaSNet 高多少） | 判定 |
|---|---|---|---|---|
| TF-GridNet 快照（第 16 个 epoch） | 0.595（+0.200） | **更强前端有明显帮助** | 0.545（+0.205） | **更强前端有明显帮助** |
| SpatialNet 快照（第 72 个 epoch） | 0.635（+0.240） | **更强前端有明显帮助** | 0.590（+0.250） | **更强前端有明显帮助** |

- 这些都只是诊断：模型还在训练中，样本只有 100 组 devel，只跑了一次，选路依赖 oracle 方位。
- SpatialNet 快照 + P2 为 0.635，与 C4-direct 的 0.670 只差 0.035。
