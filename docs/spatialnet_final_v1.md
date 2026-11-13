# SpatialNet 最终 checkpoint 评估 v1

沿用 `lora_v2_runs.md` §20–22、`frontend_compare_v1.md`、`c6_real_v1.md` 的批准配置和冻结规则。只做 devel 推理与评估，不使用 SLURP 最终 test，不修改原始 checkpoint、快照结果或既有判定。

## 训练完成确认（2026-11-07 22:33 EST）
- 训练链 21977586 → 21988892 → 21995307 → 22010857 → 22025125 全部 COMPLETED / ExitCode 0:0；最后作业于 22:33:10 结束，无新的 continuation 或同名运行/排队作业。
- `metrics.jsonl` 的 epoch 1–100 连续完整，与各段日志的 EPOCH 行逐条一致。
- `latest.pt`：epoch100、complete=True、history 与完整 metrics 一致。
- `best.pt`：ep100，完整历史最小 valid_loss = −20.314303703560604，dev SI-SDR = 20.31430916242347 dB；train_loss = −16.48988272721007。best 恰为最后一轮，但判据仍不要求 best.complete=True。
- best/latest 可完整加载，配置经 JSON 往返后与 run_config.json 一致。已有 SMS-WSJ test_summary 的 completed_epoch、selected_epoch、checkpoint SHA256、audit SHA256、目标及通道配置全部匹配本次训练。只核对原训练脚本产生的摘要来源，没有新运行任何 test。
- 完成核验脚本：`scripts/sms_wsj/spatialnet_final_check.py`。源运行目录 `<SEPARATION_REPO>/multichannel/spatialnet/exp/sms_spatialnet_small_v1/`。

## 最终模型副本
- `<SHARED_ROOT>/checkpoints/final/spatialnet_ep100_20261107-223434/best.pt`
- 14,622,162 字节；SHA256 `21caf0fc3b08cfc3b9860efbfeff9a756acecbe36101620378e47891edd76c10`。
- 复制前后源文件大小与 mtime_ns 一致，副本大小和 SHA256 与源及原训练摘要一致。旁边的 `snapshot.json` 是兼容既有推理脚本的溯源文件，明确标记 FINAL，并记录完成核验、作业链及源文件前后元数据。

## 已批准的执行配置（写于本次最终推理结果之前）
1. `scripts/sms_wsj/frontend_final.sbatch`：<SLURM_ACCOUNT> / <GPU_PARTITION>，1 A100，4 CPU，32 GB，20 分钟。顺序调用原 `frontend_snapshot.py --mode cv`、`--mode mix`；仅新增 `--final` 溯源标签与完成元数据检查，推理、形状核验和评分规则不变。cv_dev93 50 组核查未通过时不运行 mix。输出错误、形状不对或平均 direct SI-SDRi ≤0 即停止；失败不自动重试。
2. cv 核查结果放 `outputs/frontend_final/<job>/spatialnet/`；分离音频放 `<SHARED_ROOT>/data/mix_v1/separated_final/spatialnet_ep100_20261107-223434/`，新目录，不覆盖快照。
3. 分离完成且文件核验通过后，用原 `eval_snapshot_mix.py` 在 <CPU_PARTITION> 上生成 v3 选路和参考匹配。后续两路 P2 / Whisper+T2 使用批准的 30 分钟特例，其余新作业 20 分钟。
4. C5、C6δ、c6-real-v1 使用最终路径，评分、选路、噪声种子、拒答规则及一致性检查保持冻结版本。新结果与 ep72 快照并列，使用独立输出目录。
5. 在相同最终分离音频上完成 Audio LoRA C5/c6-real-v1 检查，按 §11 区分实际选中参考目标路/干扰路的实体命中；sector 不可辨惩罚与实际路由错误不混同。C3 不通过和固定 1e-4 step440 均不改变，Whisper+T2-LoRA 暂缓。

本次启动前共享存储使用率 88%，可用约 127 GB；副本约 14 MB，分离音频预计约 33 MB，无需下载或新建环境。

## 作业记录
- 前端作业 **22029593** 已提交并开始运行，节点 <NODE>，A100-PCIE-40GB。运行 commit `1815eb5`，日志确认 DIRTY 0；日志 `outputs/logs/frontend_final_22029593.out`。cv 核查和 mix 输出完成情况待核验，不预先判定通过。

## 前端与选路完成（2026-11-07）
- 22029593 COMPLETED / 0:0，1:32。cv_dev93 50 组，errors=[]，平均 direct SI-SDRi 25.93294 dB、SI-SDR 20.32118 dB；极性修正前方向匹配 0.32，修正后 1.00（排除小于15°也为1.00），PASS。两路极性修正率均1.00。mix 日志 MIX_DONE 200。
- CPU 选路作业 22029595（<SLURM_ACCOUNT>/<CPU_PARTITION>，4 CPU、16 GB、20分钟），COMPLETED / 0:0，1:05。400 个文件的8kHz、长度、单通道形状、有限值及200个样本集合核验通过，调用原 v3 函数生成选路；输出 `outputs/frontend_final_eval/ep100_20261107/spatialnet/{selection.jsonl,other_selection.jsonl,per_sample.jsonl}`。
- mix_v1 平均两人 direct SI-SDRi 18.62176 dB，目标路18.69140 dB；选路199/200=0.995，排除[0,15)后1.00，唯一错误属于[0,15)。不改变选路规则，不因结果修改模型。
- 与 ep72 并列：平均两人 SI-SDRi 18.55517 → 18.62176 dB；目标路18.58684 → 18.69140 dB；全部选路1.00 → 0.995，排除[0,15)均1.00。唯一错误 `g038_s1`（ref_stream=1，sel=0，两路估计方向差不超过5°）。这不是下游 C5/C6 的结果。

## 下游入口（写于最终下游结果之前）
- `qwen_omni_final.sbatch`：<SLURM_ACCOUNT>/<GPU_PARTITION>、1 A100、4 CPU、48GB、30分钟；选中路与另一路各一作业，调用原 `qwen_omni_snapshot.py`，输出 `outputs/qwen_omni_final/<job>_<route>/`。新参数仅标记final及提供固定LoRA适配器可选分支；基线生成参数、P2/T2、Whisper及音频处理不变。
- 两路 Audio LoRA 在基线及最终评估核验后提交：`eval_audio_lora_final.sbatch`，同样1 A100、4 CPU、48GB，20分钟，仅P2；固定 `step1/lr1e-4/ckpt_step440`，加载方式与既有第1步脚本一致。复用基线同一音频处理/解码函数；保存处理后音频SHA256及prompt，评估前必须与同路基线逐条一致。
- `eval_final_routes.py` 仅连接最终输入路径，调用原 C5/C6δ/c6-real 函数并保留一致性断言。C6δ沿用其原文档的dev半选择τ流程（只作敏感性分析）；真实C6仍固定margin τ=17°、K=8/σ=10°主条件，禁止重调。

## CPU 前置检查异常（2026-11-07，未重试）
- 回归检查 22029603（<SLURM_ACCOUNT>/<CPU_PARTITION>、4 CPU、16 GB、20分钟），使用既有 ep72 输出测试新路径入口，FAILED / 1:0，运行1秒。错误为 `ModuleNotFoundError: eval_c6_real`：新增入口未在 `python -I` 隔离模式下显式加入本目录。评分未开始，没有新的模型预测，也未覆盖任何原结果。
- 已准备最小修复：只在入口导入前将 `Path(__file__).resolve().parent` 加入 `sys.path`，与项目既有评估脚本一致。不改变实验配置、评分或冻结规则。未重提交回归作业；等待负责人批准按原CPU配置重跑。最终两路基线GPU和LoRA GPU尚未提交。
- 独立输入 dry-run 22029602（相同 CPU 配置）COMPLETED / 0:0，2:56。基线选中路、基线另一路、LoRA选中路各构造全部200条音频并测试前2条processor输入，三个 DRY_RUN_PASS。基线选中路与LoRA处理后音频SHA256为200/200逐条一致。未产生模型预测。

## 回归检查获批重跑并通过（2026-11-07）
- 经负责人批准，使用修复 commit `5022bff`，CPU 作业 **22029635** 保持4 CPU / 16 GB / 20分钟，COMPLETED / 0:0，35秒。独立目录 `outputs/final_eval_regression/ep72_retry_5022bff/`。
- 原始两路各200个样本的P2/T2/Whisper输入集合检查通过；C5评分、C6δ（除说明文字）、c6-real全部SpatialNet条件及一致性检查、风险覆盖曲线与原始ep72结果逐项完全一致。五个输出文件及SHA256记录于 `regression_check.json`，passed=true。
- 基线GPU **22029639**（selected）、**22029640**（other）按原批准配置提交，提交时工作区干净、commit5022bff。尚未判定完成。

## 最终配对分析实现说明（LoRA最终推理结果之前）
- `scripts/lora/eval_final_lora_checks.py` 复用原score/summ及真实C6选路/扇区/噪声规则，执行既定B=2000、seed2026的按夹角分层配对组bootstrap。两次方位互换和同组10次噪声重复均一并重采样；选择半/报告半沿用既有50组划分。C6主指标及拒答分母逐项断言与冻结评估输出一致。
- 同路基线/LoRA的处理后音频SHA256、prompt、样本标签和stream必须逐条相同；固定step440适配器。先使用相同ep72输入作为两方，核验全部配对差值及区间为零。
- §11按实际所选流是否等于参考目标流分组，报告预测实体总数/均数、目标/干扰者可区分值命中数、四类binding计数；噪声条件计数以10次重复的观测数为分母。sector不可辨惩罚另外保留在主指标，不将其混作实际选路错误。
- 门控仍是全200样本：C5配对翻转不低于同输入P2，真实C6主条件自然加权只含干扰不高于P2+0.01。补充诊断不改变主评分、门控或此前C3失败结论。

## 最终基线下游完成（2026-11-07）
- 22029639 selected：COMPLETED/0:0，11:49；22029640 other：COMPLETED/0:0，10:40。各P2/Whisper/T2均200条、唯一key完整、ALL_DONE。运行代码分别为5022bff/7bc26e5；所用推理与评分依赖无修改，来源记录无dirty文件。
- CPU最终评分22029669：COMPLETED/0:0，39秒；P2/WT2的真实方位一致性断言全部通过。新输出：`outputs/c5_final/ep100_20261107/eval.json`、`outputs/c6_delta_final/ep100_20261107/`、`outputs/c6_real_final/ep100_20261107/`。
- 配对诊断自检22029646：COMPLETED/0:0，4秒；相同ep72输入的全部差值和区间为零，主C6指标与冻结输出一致。自检输出 `outputs/final_eval_regression/ep72_lora_checks_selftest/checks.json`。

| C5指标 | ep72 P2 | final P2 | ep72 Whisper+T2 | final Whisper+T2 |
|---|---:|---:|---:|---:|
| target_any | 0.635 | 0.620 | 0.590 | 0.570 |
| only_interferer | 0.000 | 0.000 | 0.005 | 0.005 |
| pair_flip_accuracy | 0.330 | 0.300 | 0.320 | 0.300 |

真实C6主条件（自然夹角加权；拒答后错误率以作答样本为分母；sector主目标命中含不可辨惩罚）：

| 条件/策略 | 路线 | 目标命中 ep72→final | 只含干扰 ep72→final | final选路错误 | final覆盖率 |
|---|---|---:|---:|---:|---:|
| sector_K8_center/none | P2 | 0.54788 → 0.52519 | 0.04894 → 0.05051 | 0.13454 | 1.00000 |
| sector_K8_center/A | P2 | 0.62239 → 0.59558 | 0.00693 → 0.00924 | 0.01156 | 0.84599 |
| sector_K8_center/B | P2 | 0.63222 → 0.60732 | 0.00000 → 0.00000 | 0.00000 | 0.86155 |
| sector_K8_center/none | WT2 | 0.53253 → 0.51411 | 0.04418 → 0.04733 | 0.13454 | 1.00000 |
| sector_K8_center/A | WT2 | 0.60800 → 0.58623 | 0.00693 → 0.00924 | 0.01156 | 0.84599 |
| sector_K8_center/B | WT2 | 0.61444 → 0.59446 | 0.00000 → 0.00000 | 0.00000 | 0.86155 |
| noise_sigma10/none | P2 | 0.60012 → 0.57493 | 0.03203 → 0.03151 | 0.04880 | 1.00000 |
| noise_sigma10/A | P2 | 0.62984 → 0.59880 | 0.00276 → 0.00276 | 0.00436 | 0.85504 |
| noise_sigma10/none | WT2 | 0.58505 → 0.56671 | 0.02866 → 0.02827 | 0.04880 | 1.00000 |
| noise_sigma10/A | WT2 | 0.61775 → 0.59602 | 0.00251 → 0.00275 | 0.00436 | 0.85504 |

C6δ敏感性（全部200，等量样本，不用于更改真实C6拒答阈值）：

| δ | 路线 | 目标命中 ep72→final | 只含干扰 ep72→final | 选路错误 ep72→final |
|---|---|---:|---:|---:|
| 0 | P2 | 0.635 → 0.620 | 0.000 → 0.000 | 0.000 → 0.005 |
| 10 | P2 | 0.510 → 0.500 | 0.130 → 0.125 | 0.210 → 0.200 |
| -10 | P2 | 0.545 → 0.530 | 0.085 → 0.100 | 0.130 → 0.145 |
| 20 | P2 | 0.385 → 0.380 | 0.245 → 0.235 | 0.395 → 0.390 |
| -20 | P2 | 0.505 → 0.480 | 0.145 → 0.160 | 0.220 → 0.230 |
| 0 | WT2 | 0.590 → 0.570 | 0.005 → 0.005 | 0.000 → 0.005 |
| 10 | WT2 | 0.480 → 0.470 | 0.100 → 0.110 | 0.210 → 0.200 |
| -10 | WT2 | 0.505 → 0.480 | 0.085 → 0.090 | 0.130 → 0.145 |
| 20 | WT2 | 0.380 → 0.375 | 0.215 → 0.220 | 0.395 → 0.390 |
| -20 | WT2 | 0.460 → 0.435 | 0.135 → 0.140 | 0.220 → 0.230 |

- 最终模型的前端SI-SDRi略高，但C5理解指标未超过ep72；如实保留这一结果，不重新选择前端checkpoint。
- 固定Audio LoRA两路22029676/22029677已按原配置提交并运行；最终退化结论待输出核验。整体§6中的C3失败不变。

## 最终 Audio LoRA 检查完成
- 两路22029676/22029677均COMPLETED/0:0，各200条；音频SHA256、prompt及标签与同路基线完全相同。冻结评分22030055、配对分析22030132均COMPLETED/0:0；全部输入和主指标一致性断言通过。
- C5：目标至少命中一个0.620→0.835，配对差+0.215 [0.150,0.275]；只含干扰0→0.005；配对翻转0.300→0.630。
- 真实C6自然加权只含干扰：8扇区none 0.05051→0.05882，A 0.00924→0.00924，B 0→0；σ10 none 0.03151→0.03995，A 0.00276→0.00343。五项均通过P2+0.01点估计门控；C5翻转门控通过。
- 新增干扰命中全部在实际错路组；正确路由组干扰命中均为0。完整实体数量、binding四类、两半结果和区间见 `lora_v2_runs.md` §25。风险增长不能否认，但不支持直接归因为正确路由下空间绑定能力退化。
- 剩余SpatialNet门控全部通过，整体§6仍因既有C3失败而不通过；不重选checkpoint，不重调阈值，不启动Whisper+T2-LoRA，不使用最终SLURP test。
- LoRA新输出：`outputs/c5_final/lora_step440_20261107/`、`outputs/c6_real_final/lora_step440_20261107/`、`outputs/lora_step1_checks/final_spatialnet_ep100_20261107/`；所有原始结果保留。
