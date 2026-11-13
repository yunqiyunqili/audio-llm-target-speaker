# SLURP test 冻结文档 v1（2026-11-11 审批；尚未运行）

> - 依据：`docs/lora_slu_plan_v2.md` §9（test 只在最终冻结配置上运行一次；运行前另写文档冻结）、`docs/lora_v2_runs.md`、`docs/spatialnet_final_v1.md`、`docs/c6_real_v1.md`、`docs/mix_spec_v1.md`。
> - 本草案写作时**没有读取任何 test 音频、test 标注内容或 test_eval92 RIR**；下文 test 规模只引用 `docs/log/day1_followup4.md` §9 已记录的划分统计。
> - 由草案 `docs/slurp_test_freeze_v1_draft.md`（保留不改）定稿。草案中所有待定项由**项目负责人于 2026-11-11 决定**，见 §0；本文件提交的 commit 即冻结点。
> - 冻结后仍**不读取 test 音频**，直到项目负责人另行批准 test 执行步骤。批准后先做 CPU 构造（§4），记录 sha256，再提交 GPU 推理。test 结果出来后不再调参、不重选 checkpoint、不改阈值；重跑只按 §0-8 的失败重跑规则。

## 0. 审批决定（项目负责人，2026-11-11）
1. **音频 LoRA 列入 test。** 理由：§6 只在 C3 上不通过；C3 不是实际系统路径，实际路径（C5、c6-real）通过；不列入就无法检验预先定好的主要比较。test 上照实报告 §6 结果和 C3，作为局限。
2. **主要比较**：C1、C1-R、C5 三项（音频 LoRA − Whisper + T2-LoRA）。**Bonferroni 校正**：每项报告 98.33% 配对 bootstrap 区间（0.8333% / 99.1667% 分位数），下界 > 0 才判为显著；另报告未校正的 95% 区间。其余条件只描述。
3. **test 混合集**：按 mix_v1 规则构造 **200 组（400 个样本）**，四个夹角层等量（各 50 组）；可用配对不足 200 组时取可行的最大组数（保持四层等量），不少于 100 组；少于 100 组就停下报告。
4. **C0** 增加 T2-LoRA 在标准转写上的结果，只作描述。
5. **自然夹角加权**：主结果用 cv_dev93 分布；另报 test_eval92 自身分布的版本，只作描述。
6. **原始 Qwen 路线**用 `envs/qwen_omni`，不做两个环境的对比检查。
7. **test GPU 预算单列，上限 10 小时**；估计超过 10 小时就改为 100 组并写明。
8. **失败重跑规则**：作业因程序错误、超时或节点故障失败且没有产生完整输出时，可以用完全相同的配置重跑，并记录原因；已经产生完整输出的条件，不得因结果重跑。
9. **C5 判读口径（项目负责人 2026-11-12 确认）**：C5 主要比较按**等量（raw）口径**判读（98.33% 区间下界 > 0），自然夹角加权只作描述，与 devel 主要比较一致。

## 1. 冻结配置

| 组件 | 冻结值 |
|---|---|
| 基础模型 | Qwen2.5-Omni-7B，commit `ae9e1690`（本地快照，离线），bf16，sdpa，talker 关闭，贪心解码，`thinker_max_new_tokens=256` |
| Prompt | P2、T2（`scripts/qwen/qwen_omni_c1_p2t2.py` 的 `PROMPT_P2` / `PROMPT_T2`，原文不改）；C3 方向说明（`docs/mix_spec_v1.md`，冻结 2026-11-05） |
| ASR | Whisper large-v3，revision `06f233fe…`（`WHISPER_REV`），`language='en'`、贪心、`max_new_tokens=128` |
| 原始 Qwen | 不加适配器：原始 Qwen + P2（音频路线）；Whisper + 原始 Qwen + T2（文本路线） |
| 音频 LoRA | `checkpoints/lora_v2/step1/lr1e-4/ckpt_step440`（P2；r16/alpha32，thinker 语言模型 112 个 q/k/v/o 模块） |
| Whisper + T2-LoRA | `checkpoints/lora_v2/step2/lr1e-4/ckpt_step660`（T2） |
| 前端 | SpatialNet-small ep100 最终 checkpoint：`<SHARED_ROOT>/checkpoints/final/spatialnet_ep100_20261107-223434/best.pt`，SHA256 `21caf0fc3b08cfc3b9860efbfeff9a756acecbe36101620378e47891edd76c10` |
| 选路 | v3（`eval_c6_delta.choose`：估计方向离询问方向最近的一路，一样近取第 0 路；零时延 mic0 极性修正） |
| 拒答 | 拒答 A：margin < **τ = 17°** 拒答（固定，c6-real-v1）；拒答 B：两路估计方向都落在询问扇区内则拒答（只用于 C6-sector） |
| 评分 | slu-score-v2、binding-score-v1（normalize_v2）、c1-error-taxonomy-v1，均不修改 |
| 环境 | `<SHARED_ROOT>/envs/qwen_omni_lora`（torch 2.6.0+cu124、transformers 4.52.4、peft 0.15.2）用于 LoRA 路线；原始 Qwen 路线用 `envs/qwen_omni`（与 devel 相同，与前者只差 peft；不做两环境对比检查，§0-6） |

运行时每个作业记录：git commit（工作区必须干净）、适配器文件 SHA256、模型快照 revision、输入音频 SHA256、处理后波形 SHA256。

## 2. devel 阶段的门控事实（照实写入，不修改）
- 计划 §6 退化检查**整体不通过**，原因是 **C3**："只含干扰" P2 0.115 → 音频 LoRA 0.220（阈值 ≤ P2 + 0.02）。其余项目（C1、C4-direct、C3 目标命中、C5 配对翻转、c6-real-v1 五项）按点估计通过；c6-real-v1 8 扇区和 σ10 不拒答的"只含干扰"差值区间上界超过 0.01（`lora_v2_runs.md` §25 及 2026-11-11 §6 说明）。
- 解释（项目负责人 2026-11-11）：音频 LoRA 没有学会按方向挑人，在未分离的混合录音上更积极地抽出干扰者实体；实际系统路径（C5、c6-real）通过。
- **是否仍将音频 LoRA 列入 test：由项目负责人决定。**
  - 决定：**列入**（理由见 §0-1）；test 上照实报告 §6 结果和 C3，作为局限。
  - 决定人 / 日期：项目负责人 / 2026-11-11
- devel 上的主要比较（报告半，Audio LoRA − Whisper + T2-LoRA）：C1 +0.056 [0.018, 0.100]，C1-R +0.093 [0.044, 0.147]，C5 raw +0.090 [0.050, 0.130]。这些是 devel 结果，不作为 test 的先验门槛。

## 3. test 条件（每个条件、每条路线只运行一次）

| 条件 | 输入 | 路线 | 样本 |
|---|---|---|---|
| **C1** | split_v2 test 录音，SLURP 原始 16 kHz，不调电平 | 原始 Qwen + P2；音频 LoRA + P2；Whisper + T2；Whisper + T2-LoRA | 全部 test 录音（已记录 1,183 条 / 812 句 / 37 说话人） |
| **C0** | SLURP 标准转写（原文） | 原始 Qwen + T2；T2-LoRA + T2（仅描述，§0-4） | 812 句 / 1,183 条（按句子和按录音都报告） |
| **C1-R** | 同 C1 录音，单人完整混响：`reverb_single.reverb()` 链路，RIR 换为 **test_eval92**（分配规则同 devel：按文件名排序，`random.Random(2026).sample`，再逐条 `randrange(2)` 选 h_0/h_1） | 同 C1 四条 | 同 C1 |
| **C3** | test 混合 mic 0 单通道 + 冻结方向说明 + P2 | 原始 Qwen + P2；音频 LoRA + P2 | test 混合集全部样本 |
| **C5** | test 混合 6 通道 → SpatialNet ep100 → v3 选路（询问方向 θ1）→ `chain8(sign · out{k})`（8 kHz → `resample_poly(x,2,1)` → `level()` −30 dBFS，极性只乘一次） | 同 C1 四条；选中路与另一路都推理 | test 混合集全部样本 |
| **c6-real-v1 主条件** | 复用 C5 两路输出，只在 CPU 上改询问方向和选路：8 扇区中心划分（K=8）、σ = 10° 噪声（10 次重复，`SeedSequence([2026, σ, r])`） | 同 C5 | 同 C5 |

- 不运行：C4-direct、C6δ、K=4 / 旋转划分、σ=5°/15°（这些只在 devel 作敏感性分析）。审批决定未列入敏感性附录，test 上不报告；如需报告须另行批准，且只描述、不判读。

## 4. test 混合集构造（运行前在 CPU 上生成，与 mix_v1 相同规则）
- 语音：split_v2 test 录音（口径 A）。RIR：SMS-WSJ **test_eval92**（8 kHz，6 通道，h_0/h_1）。
- 配对过滤、抽样、混合全部沿用 `docs/mix_spec_v1.md`，只替换 split 和 RIR 集：
  1. 同性别、时长差 ≤ 1 s、每条录音最多用 3 次、D_T 和 D_I 都非空、排除共同词干 / 包含关系（`devel_capacity_mixspec_v1.py` 的规则）；
  2. 每位说话人最多 30 组、男声至少 20 组；
  3. 组数 **200 组（400 个样本）**（§0-3）；夹角四层 [0,15)、[15,30)、[30,60)、[60,180] 各 50 组，RIR 样例不重复；不可行时按 4 的倍数递减取可行最大组数（四层等量），不少于 100 组，少于 100 组停止报告；
  4. seed 2026：按 (file_a, file_b) 排序后 `random.Random(2026)` 打乱得随机排名，0/1 整数规划最大化 Σ(N_pairs − r)·x；不可行时不放宽约束，报告最接近可行结果并停止；
  5. 同一 rng 依次决定 A/B 方向、夹角层分配、层内 RIR 样例、偏移、噪声 SNR ~ U(20,30) 和噪声种子；
  6. 每组两个互换样本，询问方向都是 θ1；偏移、rir_start、early RIR（50 ms）、`extract_piece` 对齐、mic 0 early image 有效帧 RMS −30 dBFS（SIR 0 dB）、6 通道白噪声，均与 `mix_slurp_v1.py` 相同。
- 若某层 test_eval92 样例不足，停止报告，不改分层。
- 实现：现有 `make_mix_list_v1.py`、`mix_slurp_v1.py`、`reverb_single.py` 目前只接受 devel/train 与 cv_dev93/train_si284；需加入 test / test_eval92 选项（只加分支，不改已有分支的行为；用 devel 重跑核对已有清单和音频逐字节一致）。
- 生成后、任何推理前，记录并提交：清单 JSON、每个样本目录文件、C1-R 音频、manifest 的 SHA256；test 混合集的夹角分布和 test_eval92 自然夹角分布。
- **自然夹角加权**：主结果使用 c6-real-v1 已冻结的 cv_dev93 权重（78/96/155/653）；另报 test_eval92 自身分布的加权结果，仅描述（§0-5）。

## 5. 指标
- 单人条件（C1、C1-R、C0）：主指标"至少命中一个金标值"（binding-score-v1，干扰集为空，按录音）；另报意图严格准确率、slot P/R/F1、date 召回率、零实体、解析失败、c1-error-taxonomy-v1 各类计数。
- 混合条件（C3、C5、c6-real-v1）：binding-score-v1 四分类、"目标至少命中一个""目标全部命中""只含干扰"（等量与自然夹角加权）、配对翻转准确率；c6-real-v1 另报覆盖率、已回答样本的选路错误率 / 只含干扰 / 目标命中（主口径：sector 不可辨样本不计功）、风险-覆盖率曲线（τ 扫描只描述，τ = 17° 不变）。
- §11 式诊断：按实际路由正确 / 错误分组的实体数、D_T / D_I 命中数。

## 6. 主要比较、bootstrap 与判读
- **主要比较方向：音频 LoRA − Whisper + T2-LoRA**，在相同音频上配对。条件：**C1、C1-R、C5 三项共同主要**（指标均为目标至少命中一个；C5 以 raw 等量口径判读，与 devel 主要比较相同，自然加权只描述）。其他比较（只描述）：音频 LoRA − 原始 Qwen P2；Whisper + T2-LoRA − Whisper + T2；C0 原始 T2 / T2-LoRA 作为文本上限参照；C3、c6-real-v1。
- bootstrap：B = 2000，`np.random.default_rng(2026)`；主要比较报告 **98.33%**（0.8333% / 99.1667% 分位数，Bonferroni 3 项）和未校正 95%（2.5% / 97.5%）两种区间，取自同一组重抽样；其余只报 95%；
  - 单人条件：按 slurp_id 成组配对重抽样（同句录音一起抽）；
  - 混合条件：组级、按夹角层分层，同组两个互换样本一起抽；c6-real σ10 对 10 次重复的平均做 bootstrap。
- test 没有选择半 / 报告半，全部样本一起报告。
- 判读：主要比较三项各自 **98.33% 区间下界 > 0** 才判为显著；95% 区间只并列报告。devel 阶段未做校正，不追溯修改。
- 风险检查（只报告，不作门控）：C3 与 c6-real-v1 主条件上音频 LoRA 相对原始 Qwen P2 的"只含干扰"差值及区间，与 devel 的 §6 阈值并列，不因 test 结果改变 devel 判定。
- 无论结果如何，不在 test 后调整 checkpoint、阈值、prompt 或路线；异常（解析失败激增、样本数不符、哈希不一致）只报告。

## 7. 作业配置与 GPU 估计
- 所有 GPU 作业：<SLURM_ACCOUNT> / <GPU_PARTITION>，1 × A100-PCIE-40GB，4 CPU，48 GB，**20 分钟**（每个作业一个路线 × 一个分片）；SpatialNet 前端 1 A100 / 4 CPU / 32 GB / 20 分钟。CPU 构造与评分：<SLURM_ACCOUNT> / <CPU_PARTITION>，4 CPU，16 GB，20 分钟。工作区干净才提交；失败不自动重试，重跑只按 §0-8。
- 每条耗时按 devel 实测：P2 / 音频 LoRA 约 1.9–2.6 s，Whisper 约 0.32 s，T2 / T2-LoRA 约 1.1 s，模型加载约 0.5–1.5 分钟 / 作业。20 分钟内 P2 类路线约 400 条 / 片。

| 部分 | 推理次数 | 估计 GPU 时间 |
|---|---|---|
| C1（1,183 × 4 路线，Whisper 转写共享） | 4,732 | 约 2.0–2.5 h |
| C1-R（同上） | 4,732 | 约 2.0–2.5 h |
| C0（812 句 × 原始 T2 + T2-LoRA，按句推理、按录音展开） | 1,624 | 约 0.5–0.6 h |
| SpatialNet 前端（200 组 × 2 = 400 个样本） | 400 | < 0.1 h |
| C3（400 × 2 路线） | 800 | 约 0.5–0.6 h |
| C5 两路（400 × 2 路 × 4 路线） | 3,200 | 约 1.4–1.8 h |
| c6-real-v1 | 0（复用 C5，CPU） | 0 |
| 加载与分片开销（约 30–35 个作业） | — | 约 0.5–0.9 h |
| **合计**（200 组） | — | **约 7–9 GPU 小时**（上限 10 h，§0-7） |

- 预算：test 单列，上限 **10 GPU 小时**，不占 LoRA 24 h 预算。按 200 组估计约 7–9 h；执行前按实测重估超过 10 h 则改为 100 组并写明；运行中累计将超过 10 h 时停止报告。
- 顺序：CPU 构造 + sha256 记录并提交 → SpatialNet 前端 → CPU 选路 → 单人条件与 C3 → C5 两路 → CPU 评分与 bootstrap → 一次性报告。

## 运行记录

### S0：test 分支与 devel 等价核对（2026-11-12）——**未通过字面判据，已停在 S1 之前**
- 代码 commit `56731ee`（本地，**未 push**）：给构造、选路、推理、评分脚本加了 test 分支（`scripts/qwen/split_cfg.py`，默认 devel；环境变量 `SLURP_EVAL_SPLIT=test` 才切换）；新增 `merge_test_parts.py`、`score_slurp_test_v1.py`、`slurp_test_job.sbatch`、`slurp_test_s0_check.py`。
- 核对作业（<SLURM_ACCOUNT> / debug，4 CPU，16 GB，20 分钟；<CPU_PARTITION> 排队约 5 小时，改用 debug，非 test 作业）：construct 22080661 FAILED、select 22080662 FAILED、direct 22080663 PASS、score 22080664 FAILED、dryrun 22080665 PASS。归因作业 22080668（同一节点上 5cac35d 旧代码 vs 56731ee 新代码）COMPLETED。结果在 `outputs/slurp_test_v1/s0/`。
- **一致的部分**：
  - devel 混合清单逐字节一致；C1-R 428 个 16-bit 文件逐字节一致；混合 manifest 一致（除 `dir` 路径字段）；直达声参考数组和摘要一致；
  - `selection.jsonl`、`eval_snapshot_mix.json` 逐字节一致；C5 eval、c6-real（基线和 LoRA）、风险覆盖曲线、T2 配对检查、Audio−T2 配对检查都与记录逐字节一致；`--ci9833` 只增加键；
  - 推理脚本 dry-run：C5 两路处理后音频 sha256 200/200 一致（含分两片），C3 prompt 一致，条目数一致；
  - `score_slurp_test_v1.py`：devel 单人主要比较、描述指标、WER、C3 全部复现；在 devel 数据上冒烟运行，复现 C1 +0.0421 [0.0139, 0.0709]、C1-R +0.0748 [0.0376, 0.1132]、C5 +0.090 [0.05, 0.13]。
- **与记录不一致的四处**（归因作业确认：在同一节点上，**旧代码与新代码的输出逐字节相同**，所以都不是 test 分支引起的）：
  1. 混合音频 800/800 个 FLOAT wav 文件字节不同，但**样本逐位相同**；只有文件头字节 60–62 不同。这三个字节是 libsndfile 写入的 PEAK chunk 时间戳。C1-R 是 PCM_16，没有 PEAK chunk，所以逐字节一致。
  2. `per_sample.jsonl` 的诊断字段 `corr_by_stream`，189/200 行在 float64 最后 1–2 位不同（节点 / BLAS 数值差）。选路、DOA、极性和全部汇总都相同。
  3. `c6_delta_final/.../eval_c6_delta_spatialnet.json`：记录文件（11-07）缺少 `route` 键。该键由 11-09 的 commit 4580ad6 加入，其余内容相同。C6δ 不在 test 中运行。
  4. `eval_final_lora_checks.py --route-kind p2` 在当前代码上（含旧代码 5cac35d）报 `KeyError: 'C6delta_+10'`。11-09 加入的 C6δ 查询与 p2 路线的冻结一致性断言不兼容，所以 11-07 的 p2 记录无法用当前代码重现。test 分支不生成 C6δ 查询，所以 test 不受影响。

### S0 判定：**通过**（项目负责人 2026-11-12）
- **四处不一致的分类和处理**：
  - ① 混合音频文件头：**元数据**（C）。800 个 FLOAT wav 只有 PEAK 块（偏移 48 起）里的写入时间戳不同：重跑 `0x6af5d855`（2026-11-12 15:16:37 UTC），记录 `0x6aeca97b`（2026-11-05 16:06:19 UTC）。样本逐位相同，其余字节也相同（`outputs/slurp_test_v1/s0/mix_audio_samplelevel_check.json`、`attrib/attrib.json`）。接受。
  - ② 选路诊断字段：**节点浮点差异**，不影响选路。`per_sample.jsonl` 的 `corr_by_stream` 在 debug 节点 <NODE> 上 189/200 行、305 个值最后 1–2 位不同（最大差 8.9e-16）；在登录节点上新旧代码都与记录逐字节相同。选中流、DOA、极性、正确与否、`selection.jsonl` 和汇总全部相同。接受。
  - ③ C6δ 输出的 `route` 键：**元数据**（C）。11-07 的记录早于 4580ad6（11-09）加入的顶层键 `"route": "both"`，其余内容相同。接受；核对脚本按"去掉该键后完全相同"判。
  - ④ 配对检查 p2 路线：**已修复并验证**。4580ad6 加入的 C6δ 查询在已存 c6-real 结果中没有对应条件，导致 `KeyError`。最小修复（`eval_final_lora_checks.py`，+7 行）：p2 路线跳过已存 c6-real 中没有的查询，不计算，并在输出中记为 `skipped_queries`（C6delta_±10、±20）；其他计算不变。修复后在 devel 上重跑，去掉元数据键 `route_kind`（4580ad6 加入）和 `skipped_queries` 后，与 11-07 的记录**逐字节相同**。记录生成时的代码版本 0803c3f 也逐字节重现该记录。test 分支本来就不生成 C6δ 查询。
- **② 对 test 的影响**：test 在计算节点上运行，`per_sample.jsonl` 里的诊断相关系数可能在最后一位与登录节点不同。它不进入选路、任何指标或判定。
- **S5 评分脚本 devel 全量重跑**：用修复后的当前代码，在登录节点运行 `slurp_test_s0_check.py score`（`outputs/slurp_test_v1/s0/score_r3_login/check.json`），**17/17 通过**：
  - `eval_final_routes.py` 基线与 LoRA：C5、c6-real、风险覆盖曲线逐字节相同；C6δ 按 ③ 判相同；
  - `eval_final_lora_checks.py` p2（按 ④）、t2、audio_vs_t2：数值相同；`--ci9833` 只增加区间键；
  - `merge_test_parts.py` 分两片再合并：与原文件逐字节相同；
  - `score_slurp_test_v1.py`：devel 复现检查通过；主流程在 devel 输入上复现 C1 +0.0421 [0.0139, 0.0709]、C1-R +0.0748 [0.0376, 0.1132]、C5 +0.090 [0.05, 0.13]。
  - `--natural test_eval92` 只允许在 test 上运行（脚本断言），devel 上无法重跑；它只改变权重向量，其余代码路径与上述相同。
- 先前 dry-run（22080665）和直达声参考（22080663）通过；清单、C1-R 和选路结果见上节。
- 重复 commit 处理：重复运行产生的提交 202c6fa 已另存为备份分支，master 回退到 8b90d43。其中经核实属实的信息已并入本节和 FSC 文档：登录节点上 `per_sample` 可复现；`score_v2_r2/` 与 `score_v2/` 逐字节相同；`per_item.jsonl` 的 sha256。
- **S1 开始后不再修改任何评分或构造代码。**

### S1：test 构造（2026-11-12，<SLURM_ACCOUNT> / <CPU_PARTITION>，4 CPU，16 GB，20 分钟；代码 b8b41d7，工作区干净）
| 作业 | 内容 | 状态 / 用时 |
|---|---|---|
| 22080676 | 混合清单 `make_mix_list_v1.py --v2 --split test --n 200` | COMPLETED 0:16 |
| 22080677 | C1-R `reverb_single.py --split test`（test_eval92） | COMPLETED 1:46 |
| 22080730 | 混合音频 `mix_slurp_v1.py`（代码 d72e5c0） | COMPLETED 1:07 |
| 22080731 | 直达声参考 `direct_refs_mix_v1.py`（sms_wsj 环境） | COMPLETED 1:46 |
- **混合清单** `configs/mix_v1_test_v1.json`，sha256 `26c0fb57e79fa4442526c824cd1d2c79e558af727dc1c11f41e4dea26b02695f`，已在 d72e5c0 提交。
  - 规模：可用配对 1,396 个；200 组的整数规划一次可行（HiGHS Optimal），**不需要减组**。
  - 四个夹角层各 50 组；200 个 test_eval92 RIR 样例互不重复。
  - 用到 281 条录音（每条用 1/2/3 次的录音分别为 184/75/22 条）、34 个说话人；单个说话人最多 30 组；男声 63 组（要求至少 20 组）。
  - T60 0.204–0.500 s；两人时长差 ≤ 0.96 s。
- **混合音频** `data/mix_v1/test_seed2026`：400 个样本、1,600 个 wav，manifest sha256 `a70b7e46…dfdd`。
  - SIR 0 dB（偏差 < 5e-15）；SNR 19.76–30.12 dB；rir_start 18–43。
  - 观测信号峰值最大 1.185，5 个样本的峰值 ≥ 1；绝对值 ≥ 1 的采样点数：观测 12 个、mic0_16k 2 个。文件存为 FLOAT，不截断，只记录、不调增益（同 mix_v1 规则）。
- **C1-R** `data/lora_v1/test_rev_eval92`：1,183 个 16-bit wav，与 split_v2 test 录音集合一致；1,183 个 test_eval92 样例互不重复；manifest sha256 `d38900c8…a2da`。
  - **削波 1 条**：`audio-1490703179-headset.flac`（样例 607），电平调整后峰值 1.054，写 16-bit 时 2 个样本被截断。
  - 按 train 先例（§15-5）**不重新生成**，照实记录。
- **直达声参考** `outputs/slurp_test_v1/direct_refs_mix`：400 个 npz，摘要 sha256 `bb421901…2dbc`。
  - cv_dev93 校验最大差 1.44e-8，通过；early image 重算最大差 5.0e-8，通过。
- 全部 3,188 个文件的 sha256：`configs/slurp_test_v1_s1_sha256.txt`（该文件 sha256 `d6475d2e…d972`）。
- **夹角分布**：test 混合集四层各 50 组。test_eval92 自身的自然分布为 104 / 94 / 207 / 927（共 1,332），只用于描述性加权；主结果仍用 cv_dev93 的 78 / 96 / 155 / 653。

### S2：SpatialNet 前端与选路（2026-11-12；代码 9c4bd5b）
| 作业 | 内容 | 资源 | 状态 / 用时 |
|---|---|---|---|
| 22080732 | `frontend_snapshot.py --final --mode cv`（cv_dev93 50 组核查） | <GPU_PARTITION> 1×A100 / 4 CPU / 32 GB / 20 min | COMPLETED 1:04 |
| 22080733 | `--mode mix`（test 400 个样本 → `data/mix_v1/separated_final_test/spatialnet_ep100_20261107-223434/`） | 同上 | COMPLETED 0:41 |
| 22080734 | `select_final.py`（v3 选路 + 另一路） | <CPU_PARTITION> / 4 CPU / 16 GB / 20 min | COMPLETED 1:16 |
- **cv 核查 PASS**：errors=[]；direct SI-SDRi 25.933 dB，SI-SDR 20.321 dB；极性修正前选路 0.32，修正后 1.00。与 devel 阶段的 22029593 一致。
- **分离**：400 个样本 × 2 路，8 kHz，形状和有限值检查通过（MIX_DONE 400）；manifest sha256 `8301ebdc…1a50`。
- **选路**（只作描述）：
  - test 两人平均 direct SI-SDRi 18.29 dB，目标路 18.36 dB；
  - 选路准确率：全部 0.980，排除 [0,15) 后 0.997；
  - 分层：[0,15) 0.93、[15,30) 0.99、[30,60) 1.00、[60,180] 1.00；排除 [0,15) 后唯一的错误是 `g162_s2`；
  - 极性翻转率：路 0 为 1.00，路 1 为 0.9975。
  - `selection.jsonl` sha256 `ed39744b…5c69`，`other_selection.jsonl` `d40382d6…fe9`，`per_sample.jsonl` `9e4a8b9f…c255`。
- **GPU 重估**（按 §0-7；用 devel 实测的每条耗时）：P2 1.20 s，音频 LoRA 2.05 s，Whisper 0.23 s，T2 1.18 s，T2-LoRA 1.9–2.2 s，C0 1.22 s，C5 每路 P2+W+T2 2.86 s；每个作业加载约 1.6 min。
  - S3 + S4 共 47 个 20 分钟作业，推理约 7.4 h，加载约 1.25 h，合计约 **8.7 GPU 小时**；加上 S2 的 0.03 h，**不超过 10 小时，保持 200 组**。
- **分片**（每片估计 ≤ 14 分钟）：
  - C1、C1-R 各：P2 2 片、Whisper+T2 3 片、音频 LoRA 4 片、T2-LoRA 4 片；
  - C0：T2 2 片、T2-LoRA 4 片；
  - C3：原始 P2 1 片、音频 LoRA 2 片；
  - C5 每路（选中路 / 另一路）：基线 2 片、音频 LoRA 2 片、T2-LoRA 2 片。

### S3 / S4：推理（2026-11-12 11:19–12:23 EST；代码 3899ffd，所有作业日志都记录同一 commit，工作区干净）
- 共 49 个作业：47 个 GPU 作业（<SLURM_ACCOUNT> / <GPU_PARTITION>，1×A100，4 CPU，48 GB，20 分钟）和 2 个 CPU 合并作业（<CPU_PARTITION>）。清单见 `outputs/slurp_test_v1/runs/jobs.tsv`（作业号 22080745–22080793）。
- 环境：原始 Qwen 路线用 `envs/qwen_omni`；音频 LoRA 和 T2-LoRA 路线用 `envs/qwen_omni_lora`。
- **结果**：49/49 COMPLETED 0:0；**没有失败，也没有重跑**。每个作业结束后都用 sacct 和输出核对：条数等于分片应有的条数，`file` 不重复，每行都是合法 JSON。最长的作业是 c1_audio_2，用时 14:21。

| 条件 | 路线（作业数） | 条数 | 解析状态（非 ok） |
|---|---|---|---|
| C1 | 原始 P2（2）/ Whisper+T2（3）/ 音频 LoRA（4）/ T2-LoRA（4） | 各 1,183 | P2 bad_json 1；T no_json 3；音频 LoRA 0；T2-LoRA 0 |
| C1-R | 同上 | 各 1,183 | P2 0；T no_json 6；音频 LoRA 0；T2-LoRA bad_json 1 |
| C0 | 原始 T2（2）/ T2-LoRA（4） | 各 1,183 | T2 no_json 5、bad_json 1；T2-LoRA 0 |
| C3 | 原始 P2（1）/ 音频 LoRA（2） | 各 400 | P2 no_json 1；音频 LoRA 0 |
| C5 选中路 | 基线 P2+W+T2（2）/ 音频 LoRA（2）/ T2-LoRA（2） | 各 400 | P2 no_json 1、T2 no_json 3；音频 LoRA 0；T2-LoRA 0 |
| C5 另一路 | 同上 | 各 400 | P2 0、T2 no_json 3；音频 LoRA 0；T2-LoRA 0 |
- 表中不含 W（Whisper 转写）：它本来就不是 JSON，解析状态都记为 no_json，与 devel 相同。
- 合并作业：22080784（选中路基线）和 22080791（另一路基线），由 `merge_test_parts.py` 合并为各 400 条，sha256 记在各自的 `merge.json`。
- **GPU 用量**：S3 + S4 共 7.803 GPU 小时；加上 S2 的 0.029 h，test 合计 **7.83 GPU 小时**，在 10 小时上限内（估计为 8.7 h）。

### S5：评分与一次性报告（2026-11-12；代码 0ce49eb；<SLURM_ACCOUNT> / <CPU_PARTITION>，4 CPU，16 GB，20 分钟）
- **作业**（全部 COMPLETED 0:0，没有重跑）：
  - 合并 22081838–41（C5 音频 LoRA、T2-LoRA 两路，各 400 条）；
  - `eval_final_routes.py`：基线 22081842、音频 LoRA 22081843，以及 test_eval92 加权版 22081844、22081845；一致性检查（询问方向 θ1 时与 C5 评分相同）全部通过；
  - `eval_final_lora_checks.py`：p2 22081846、t2 22081849、audio_vs_t2 22081850、audio_vs_t2 的 test_eval92 加权版 22081851；冻结指标断言全部通过；
  - 报告 22081888：`score_slurp_test_v1.py`，输入 `outputs/slurp_test_v1/score/inputs.json`，输出 `outputs/slurp_test_v1/score/report/report.json`（sha256 `e457aa3cbdd16a6ec68bfc9204671cda811d2b488ce983c540e074a4d994db8f`）。
- 规则全部按冻结版本执行，没有因结果改动任何规则。

#### 主要比较（音频 LoRA − Whisper + T2-LoRA，target-any，配对；Bonferroni 3 项）
| 条件 | n | 音频 LoRA | Whisper+T2-LoRA | 差值 | 95% 区间 | **98.33% 区间** | 显著（98.33% 下界 > 0） |
|---|---|---|---|---|---|---|---|
| C1 | 1,183 条 / 812 句 | 0.8478 | 0.8149 | **+0.0330** | [+0.0110, +0.0545] | **[+0.0064, +0.0595]** | 是 |
| C1-R | 1,183 / 812 | 0.8056 | 0.7506 | **+0.0549** | [+0.0307, +0.0795] | **[+0.0247, +0.0858]** | 是 |
| C5（raw，等量） | 400 个样本 / 200 组 | 0.7550 | 0.6825 | **+0.0725** | [+0.0349, +0.1125] | **[+0.0250, +0.1200]** | 是 |
- 只作描述：C5 自然夹角加权（cv_dev93）+0.0575，95% [−0.0088, +0.1292]，98.33% [−0.0230, +0.1450]。

#### 描述性结果（不判读）
- **单人条件**：

| 条件 / 路线 | target-any | 意图准确率 | 实体 F1（严格） | 只比值 F1 | date 召回 | 解析失败 | 零实体 |
|---|---|---|---|---|---|---|---|
| C1 原始 P2 | 0.567 | 0.696 | 0.324 | 0.506 | 0.288 | 1 | 45 |
| C1 Whisper+T2 | 0.552 | 0.668 | 0.324 | 0.491 | 0.313 | 3 | 41 |
| C1 音频 LoRA | 0.848 | 0.909 | 0.743 | 0.779 | 0.850 | 0 | 0 |
| C1 Whisper+T2-LoRA | 0.815 | 0.877 | 0.717 | 0.754 | 0.854 | 0 | 0 |
| C1-R 原始 P2 | 0.529 | 0.672 | 0.304 | 0.476 | 0.279 | 0 | 41 |
| C1-R Whisper+T2 | 0.499 | 0.624 | 0.302 | 0.453 | 0.317 | 6 | 66 |
| C1-R 音频 LoRA | 0.806 | 0.884 | 0.703 | 0.742 | 0.829 | 0 | 0 |
| C1-R Whisper+T2-LoRA | 0.751 | 0.843 | 0.661 | 0.693 | 0.825 | 1 | 1 |
| C0 原始 T2（标准转写） | 0.621 | 0.708 | 0.354 | 0.558 | 0.325 | 6 | 36 |
| C0 T2-LoRA（标准转写） | 0.868 | 0.904 | 0.766 | 0.805 | 0.888 | 0 | 0 |

  - Whisper WER：C1 0.074，C1-R 0.140。
  - C0 中同一句子的不同录音输出完全相同，两条路线都是 0 句有差异。
- **其他配对比较**（95% 区间）：
  - 音频 LoRA − 原始 P2：C1 +0.281 [0.240, 0.319]，C1-R +0.276 [0.237, 0.313]，C5 +0.260 [0.205, 0.313]；
  - T2-LoRA − Whisper+T2：C1 +0.263 [0.222, 0.302]，C1-R +0.252 [0.214, 0.288]，C5 +0.258 [0.203, 0.310]。
- **C5**（400 个样本）：

| 路线 | target-any | 只命中目标 | 只命中干扰者 | 配对翻转 |
|---|---|---|---|---|
| 原始 P2 | 0.495 | 0.433 | 0.0025 | 0.250 |
| Whisper+T2 | 0.425 | 0.368 | 0.0025 | 0.165 |
| 音频 LoRA | 0.755 | 0.740 | 0.015 | 0.555 |
| Whisper+T2-LoRA | 0.683 | 0.668 | 0.0175 | 0.475 |

  - Whisper 相对目标句的 WER 为 0.177。
- **c6-real-v1 主条件**（自然加权 cv_dev93；括号内为 Audio − T2-LoRA 的等量配对差 [95%]）：

| 条件 / 拒答 | 覆盖率 | target-any：P2 / WT2 / 音频 LoRA / T2-LoRA | 只命中干扰者：P2 / 音频 LoRA / T2-LoRA |
|---|---|---|---|
| 8 扇区 / 不拒答 | 1.000 | 0.450 / 0.384 / 0.696 / 0.650（+0.040 [0.008, 0.075]） | 0.026 / 0.050 / 0.049 |
| 8 扇区 / 拒答 A | 0.829 | 0.521 / 0.445 / 0.803 / 0.746（+0.072 [0.014, 0.131]） | 0.0035 / 0.014 / 0.015 |
| 8 扇区 / 拒答 B | 0.872 | 0.516 / 0.441 / 0.798 / 0.745（+0.062 [0.011, 0.114]） | 0.0011 / 0.012 / 0.011 |
| σ10 / 不拒答 | 1.000 | 0.493 / 0.426 / 0.758 / 0.704（+0.062 [0.026, 0.098]） | 0.024 / 0.048 / 0.045 |
| σ10 / 拒答 A | 0.855 | 0.520 / 0.451 / 0.804 / 0.750（+0.065 [0.011, 0.119]） | 0.0022 / 0.011 / 0.012 |

  - test_eval92 自身分布加权的版本（只作描述）数值接近，见 `report.json` 的 `c6_real_test_eval92_descriptive`。
- **风险检查**（只报告，不作门控；与 devel §6 阈值并列）：
  - C3（单通道混合 + 方向说明）：只命中干扰者 P2 0.128 → 音频 LoRA 0.195（+0.068，超过 devel 阈值 +0.02）；target-any 0.118 → 0.240。按 devel §6 规则不通过，与 devel 结论相同。
  - c6-real 自然加权的"只命中干扰者"，音频 LoRA − P2：8 扇区不拒答 +0.024、拒答 A +0.010、拒答 B +0.011，σ10 不拒答 +0.024、拒答 A +0.009。按 devel 阈值（≤ +0.01），只有 σ10 拒答 A 在阈值内。
  - C5 配对翻转 0.250 → 0.555（不低于 P2）。
  - Whisper+T2-LoRA 的"只命中干扰者"与音频 LoRA 相近（8 扇区不拒答 0.049 vs 0.050），说明这一增量来自 LoRA 抽取的实体更多，并非音频路线特有。
  - §11 实际路由诊断：
    - 路由正确时，干扰者值命中很少：C5 中 392 次观测仅 3 次；σ10 中 3,420 次观测 27 次。
    - 路由错误时，LoRA 命中干扰者值比 P2 多：8 扇区 44 vs 26，σ10 464 vs 270。
- 以上结果出来后，没有调整任何 checkpoint、阈值、prompt 或路线。

#### test 局限（项目负责人 2026-11-12 决定照实写明；数字来自 `outputs/slurp_test_v1/score/`，sha256 见 `docs/slurp_test_report_v1.md` 附录）
- **devel §6 阈值在 test 上只作描述**，不构成 test 的门控，也不改变 devel 的判定。
- **C3**（单通道未分离混合 + 方向说明；`report.json` → `mix.C3`）："只命中干扰者"原始 P2 **0.128 → 音频 LoRA 0.195**（+0.068；devel 阈值 +0.02）；target-any 0.118 → 0.240。说明音频 LoRA 不会按方向挑人，在未分离的混合录音上会更积极地抽出干扰者实体。
- **真实系统路径**（c6-real-v1 主条件，cv_dev93 自然加权；`checks_p2/checks.json`、`checks_t2/checks.json`）："只命中干扰者"的绝对值，以及两种 LoRA 相对各自基线的配对差 [95%]：

| 条件 / 拒答 | 覆盖率 | 原始 P2 | 音频 LoRA | 音频 LoRA − P2 | 原始 Whisper+T2 | Whisper+T2-LoRA | T2-LoRA − Whisper+T2 |
|---|---|---|---|---|---|---|---|
| 8 扇区 / 不拒答 | 1.000 | 0.0260 | 0.0500 | +0.0241 [+0.0106, +0.0421] | 0.0268 | 0.0486 | +0.0219 [+0.0081, +0.0395] |
| 8 扇区 / 拒答 A | 0.829 | 0.0035 | 0.0139 | +0.0104 [0.0000, +0.0292] | 0.0035 | 0.0147 | +0.0111 [0.0000, +0.0292] |
| 8 扇区 / 拒答 B | 0.872 | 0.0011 | 0.0117 | +0.0106 [−0.0011, +0.0291] | 0.0011 | 0.0106 | +0.0094 [−0.0022, +0.0266] |
| σ10 / 不拒答 | 1.000 | 0.0241 | 0.0479 | +0.0238 [+0.0127, +0.0403] | 0.0216 | 0.0449 | +0.0233 [+0.0122, +0.0388] |
| σ10 / 拒答 A | 0.855 | 0.0022 | 0.0111 | +0.0089 [+0.0006, +0.0250] | 0.0025 | 0.0122 | +0.0098 [−0.0001, +0.0263] |
  - 拒答 B 只用于扇区条件。覆盖率是自然加权值，四条路线相同（拒答只依赖选路，与路线无关）。
- **按实际路由正确 / 错误拆开的命中次数**（不拒答；观测次数 / 目标值命中 / 干扰者值命中）：

| 条件 | 路由 | 原始 P2 | 音频 LoRA | 原始 Whisper+T2 | Whisper+T2-LoRA |
|---|---|---|---|---|---|
| C5（询问方向 = θ1） | 正确（392） | 202 / 0 | 327 / 3 | 175 / 0 | 295 / 3 |
| C5 | 错误（8） | 0 / 1 | 0 / 3 | 0 / 1 | 0 / 4 |
| 8 扇区 | 正确（337） | 179 / 0 | 285 / 3 | 150 / 0 | 258 / 3 |
| 8 扇区 | 错误（63） | 0 / 26 | 0 / 44 | 0 / 27 | 0 / 42 |
| σ10（10 次重复） | 正确（3,420） | 1,757 / 0 | 2,848 / 27 | 1,528 / 0 | 2,578 / 27 |
| σ10 | 错误（580） | 0 / 270 | 0 / 464 | 0 / 239 | 0 / 425 |
  - 表中每格为"目标值命中 / 干扰者值命中"，括号内为观测次数。
- **说明**：
  - "只命中干扰者"的增量在两种 LoRA 上相近（8 扇区不拒答：+0.024 对 +0.022；σ10：+0.024 对 +0.023）。它来自 LoRA 抽实体更积极：路由正确时干扰者值几乎不被命中，选错路时更容易把错误那一路里的干扰者实体抽出来。**这不是音频路线特有的。**
  - 拒答（A 或 B）可将增量降到约 +0.01（+0.009 至 +0.011），但区间上界仍约为 +0.025 至 +0.029。
- **主要比较的边界**：
  - C1 的 98.33% 区间下界为 **+0.006**，接近 0；
  - C5 的自然加权版本（cv_dev93）+0.057，95% [−0.009, +0.129]，98.33% [−0.023, +0.145]，**跨 0，只作描述**；主判读用等量（raw）口径（§0-9）。

### 复现核对（2026-11-13，CPU，登录节点）
- 用 `scripts/lora/score_slurp_test_v1.py --inputs outputs/slurp_test_v1/score/inputs.json` 从已有输出重算，三项主要比较的点估计、95% 和 98.33% 区间都与报告逐位一致，**report.json 逐字节相同**（sha256 `e457aa3c…db8f`）。
- 流程中的 30 个脚本及其递归引用的本地模块（共 64 个文件）都已提交，没有引用未跟踪的代码。
  - 引用的数据清单 `data/slurp/manifests/split_v2_seed2026/*.csv` 不入库（`data/` 在 `.gitignore` 中），可由 `scripts/slurp/make_slurp_split_v2.py` 重建，sha256 见 README。
- git 跟踪的 `outputs/` 文件为 0 个；没有被跟踪的音频或权重（wav、flac、pt、safetensors、npz 等）。最大的被跟踪文件是 `configs/fsc_ext_v1_list.json`（1.27 MB）。

### 划分清单重建核对（2026-11-13，CPU；项目负责人决定方案 (b)）
- **作业**：<CPU_PARTITION> 22092059（2 CPU、8 GB、40 分钟），在 HEAD 89b1f06 的导出副本中运行 `scripts/slurp/make_slurp_split_v2.py`。输入为 `data/slurp/annotations/8eb1654` 和 `results/slurp/headset_durations.csv`（sha256 `0af9a403…e04e`）。
  - 先提交的 22092056 因导出副本中缺少这个时长表（当时未入库），已取消。
- **保留目录**：`<SHARED_ROOT>/tmp_split_rebuild/`。用途是重建核对的证据，不作为数据来源：
  - `repo/` 是导出的代码副本，其中的 `configs/`、`results/` 已被脚本改写；
  - `out_manifests/` 是重建出的六个清单文件。
- **结果**：
  - 三个 `*_recordings.csv` 与原文件逐字节相同；
  - 三个 `*_candidate_pairs.csv.gz` 解压后逐字节相同，文件 sha256 只因 gzip 文件头中的写入时间戳不同（原 2026-11-05，重建 2026-11-13）而不同；
  - `configs/slurp_split_v2_seed2026.json` 只差标注目录的路径字段（相对 / 绝对）；
  - `results/slurp/split_v2_stats.json` 只差两个总变差距离的第 17 位有效数字和路径字段。
- **入库**：`data` 是指向共享存储的符号链接，git 不能跟踪其中的文件。因此六个**原文件**（含 2026-11-05 的 `.csv.gz`）以逐字节副本形式入库到 `manifests/slurp_split_v2_seed2026/`；`results/slurp/headset_durations.csv` 从 `.gitignore` 中移除并入库。sha256（含解压后内容）见 README。
