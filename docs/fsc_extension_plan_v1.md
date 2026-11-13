# FSC 扩充方案 v1（2026-11-12 审批；冻结点 = 提交本文件、`configs/fsc_mapping_v1.json` 和 `scripts/fsc/score_extension_v1.py` 的 commit）

> - 目的：把 SLURP test 的主要比较（音频 LoRA − Whisper + T2-LoRA）在另一个数据集上重复，样本量足以给出确定的区间。
> - **不使用 FSC official test**；SLURP test 不再运行。
> - 模型、prompt、前端、选路、拒答、checkpoint 全部沿用 SLURP test 冻结配置（`docs/slurp_test_freeze_v1.md` §1），不重新训练、不调参。
> - **冻结后不再修改映射表和评分代码。**
> - 原来的 64 条 pilot 用新映射重算，**只作探索性分析**，与扩充结果分开报告，也不用于修改映射或规则。

## 0. 审批决定（项目负责人，2026-11-12）
1. **单人样本**：1,240 条（248 句 × 5 条，seed 2026）；bootstrap 按句子文本成组重抽样。
2. **空间条件**：360 组（720 个样本），RIR 层内不重复，与 SLURP test 规则一致。
3. **主指标** target-any；clean、reverb、spatial 三项用 Bonferroni 98.33% 区间，与 SLURP test 相同。intent_exact 和 intent_family 只作描述。
4. **加入 C0**（标准转写 + T2 和 T2-LoRA）和原始 Whisper + T2 路线，只作描述。
   - 总 GPU 预算单列 **12 小时**；
   - 重估超过 12 小时，就先去掉原始 Whisper + T2 的单人条件，并写明。
5. **说话人约束**：每人不超过组数的 25%，男声组至少 10%。
6. **映射表**（§2）：
   - 每条规则注明依据：(a) 标签定义，或 (b) FSC train 标注中的频次，并给出计数；
   - 不以任何模型输出为依据；
   - mute 按 FSC train 中含 "mute" 的句子最常见的动作标签映射；
   - wemo 只映射动作，不给默认对象；
   - heat 同义词只用 FSC train 中对象为 heat 的句子里实际出现的说法。
7. **pilot 64 条**：用新映射重算，只作探索性分析，与扩充结果分开报告。
- **停止条件**：
  - 抽样后可用组数少于 300 组；
  - 任何作业重跑后仍失败；
  - GPU 重估超过 12 小时，且去掉原始 Whisper + T2 单人条件后仍超过。
  - 失败重跑同 SLURP §0-8：没有完整输出时才可原样重跑一次。

## 1. 样本

### 1.1 单人条件（clean、reverb）
- **来源**：FSC valid（3,118 条，10 位说话人，248 句文本，31 个三元组；`valid_data.csv` sha256 `5ac95de1…5197`）。排除 pilot 已用的 64 条（`outputs/fsc_pilot/valid64_manifest.jsonl`，sha256 `64a695df…c173`，按 path 匹配），剩 3,054 条。
- **抽样规则**（运行前写定）：
  1. 248 句文本按文本字符串排序；`rng = random.Random(2026)`。
  2. 对每句文本：把剩余录音按 path 排序后用 rng 打乱；按打乱顺序逐条选入，每位说话人只选一次，直到选满 5 条。
     - 剩余录音不足 5 位不同说话人时，再按同一顺序补足。核查：每句文本至少有 6 条剩余录音、至少 5 位不同说话人，所以总能选满。
  3. 共 **248 × 5 = 1,240 条**。31 个三元组的分布与 valid 的文本分布成比例。
- **与 SLURP train 的文本重叠**：pilot 审计发现 valid 中有 349 条录音（25 句文本）与 SLURP train 的文本完全相同，另有 15 条近似重复（`outputs/fsc_pilot/audit.json`）。
  - **不排除**这些录音，逐条标记；
  - 另报一个敏感性结果：去掉这 25 句文本后重算，只作描述。
- **样本数的理由**：
  - pilot 上 Audio − T2-LoRA 配对差的逐条标准差为 0.33（clean）、0.37（reverb）、0.36（spatial）。只用于估方差，不用于映射。SLURP test 上 C1 用 1,183 条录音，98.33% 区间半宽约 0.027。
  - bootstrap 按**文本**成组（相当于 SLURP 的 slurp_id），同一文本的多条录音之间相关。设逐条标准差 σ = 0.32 至 0.40、文本内相关 ρ = 0 至 0.3，98.33% 区间的预期半宽如下：

| σ | ρ | 1,240 条（每句 5 条） | 3,054 条（全部） |
|---|---|---|---|
| 0.32 | 0 / 0.1 / 0.3 | 0.022 / 0.026 / 0.032 | 0.014 / 0.020 / 0.029 |
| 0.40 | 0 / 0.1 / 0.3 | 0.027 / 0.032 / 0.040 | 0.017 / 0.025 / 0.036 |

  - FSC 只有 248 句文本，文本内相关较高时，增加录音的收益很快递减。1,240 条在 ρ ≤ 0.1 时约为 ±0.02–0.03，只有全部录音的 40%。
  - **采用 1,240 条**（§0-1）。

### 1.2 两人空间条件（spatial）
- **配对**：从上述 1,240 条录音中按 mix_v1 风格的规则抽取：
  - 同性别（FSC valid 中女 7 位、男 3 位）、不同说话人、时长差 ≤ 1 s；
  - 目标与干扰者的三元组值（action、object、location，去掉 `none`）互有可区分的值，即 D_T 和 D_I 都非空；
  - 每条录音最多用 3 次。
- **抽样**：seed 2026。按 (path_a, path_b) 排序后打乱，再用 0/1 整数规划按排名选出 N 组；不可行时按 4 的倍数递减（与 SLURP test 相同）。
- **说话人约束**：FSC 只有 10 位说话人，SLURP 的"每位说话人 ≤ 30 组"按比例无法满足。改为"每位说话人 ≤ 组数的 25%"、男声组 ≥ 组数的 10%。已批准（§0-5）。
- **RIR**：SMS-WSJ **test_eval92**（与 SLURP test 一致，不用 devel 调参时用过的 cv_dev93），四个夹角层等量，同一 FSC 集合内 RIR 样例不重复。与 SLURP test 用过的样例允许重复（RIR 是模拟抽样，不涉及调参）。
- **混合**：偏移、早期 RIR、电平、SIR 0 dB、SNR U(20, 30) dB 与 `mix_slurp_v1.py` 相同；每组交换目标得到两个样本；询问方向为 θ1。
- **组数**：
  - SLURP test 的 C5 用 400 个样本，98.33% 半宽约 0.048。要到 ±0.03，约需 1,000 个样本（500 组）。
  - 但 test_eval92 的 [15,30) 层只有 94 个样例，**RIR 不重复时四层等量最多 94 × 4 = 376 组**。
  - **采用 360 组（720 个样本，每层 90 组）**，RIR 层内不重复（§0-2）；预期半宽约 0.035。整数规划不可行时按 4 递减，少于 300 组就停下。

## 2. 映射表（冻结文件 `configs/fsc_mapping_v1.json`；执行代码 `scripts/fsc/score_extension_v1.py` 的 `predict()`）

- **说明**：映射表起草人曾在 pilot 的评分和抽查中看过 pilot 的部分模型输出。映射表的规则**只依据**下面两类来源，没有以任何模型输出为依据，也没有按模型输出调整：
  - **(a) 标签定义**：SLURP 训练集标签清单 `configs/slurp_label_inventory_train.json`（60 个意图、55 个实体类型，sha256 `a3e86457fafa4aaf60f28654f3778a278b29132068977285b3996c46773880e9`）中意图和实体类型的名称，以及 FSC 的 31 个 (action, object, location) 三元组；
  - **(b) FSC train 标注中的频次**：`train_data.csv` 共 23,132 条。计数 = 标注文本中含该词的 train 记录数，以及它们落在各标签上的分布。
- **适用对象**：只作用于模型输出解析后的 `intent` 和 `entities`。每个实体值先做 `normalize_v2`，再按 `toks` 切成整词，在值内整词匹配；按输出顺序取第一个命中的实体。**不使用 transcript 字段。**
- **单元测试**：15 个人工构造的输出（不是模型输出）全部按规则得到预期三元组。

### 2.1 对象 object
- **收词规则**：词在 FSC train 中至少 95% 的记录落在同一个非 none 对象类，且至少 20 条记录，并且是对象本身或其程度的名称。动词和功能词不收，完整列表连同计数见 json 的 `object_keywords_rejected`。

| FSC object | 关键词 | 依据 | 计数（含该词的 train 记录 / 落在该对象的记录） |
|---|---|---|---|
| lights | lights | a + b | lights 4,794 / 4,794。单数 light 在 train 中出现 0 次，不收 |
| lamp | lamp | a + b | 792 / 792 |
| heat | heat, temperature, heating, hotter, cooler | b（只取对象为 heat 的句子里实际出现的说法） | heat 2,946 / 2,946；temperature 2,805 / 2,805；heating 1,322 / 1,322；hotter 92 / 92；cooler 90 / 90 |
| volume | volume, sound, louder, loud, quiet, quieter, audio, softer | a + b | 2,203；647；492；476；278；192；190；100（各自 100% 落在 volume） |
| juice / newspaper / socks / shoes | 同名词 | a + b | 466 / 551 / 538 / 536（各 100%） |
| Chinese / English / German / Korean | chinese / english / german / korean | a + b | 449 / 349 / 342 / 345（各 100%） |
| music | （无关键词） | b | music 一词 988 条落在 music、198 条落在 volume，占 83% < 95%，不收。对象 music 只来自 play_music 的默认对象 |
- 同一实体值中同时有 lights 和 lamp 的关键词时取 lamp。依据 a：FSC 没有同时含两者的三元组。
- 都没命中时用意图的默认对象（§2.3）；仍没有就为 `none`。

### 2.2 地点 location
| FSC location | 关键词 | 依据 | 计数 |
|---|---|---|---|
| kitchen | kitchen | a + b | 2,608 / 2,608 |
| bedroom | bedroom | a + b | 2,412 / 2,412 |
| washroom | washroom, bathroom | a + b | washroom 2,211 / 2,211；bathroom 1,751 / 1,751（FSC 标签均为 washroom） |
| none | 都没命中 | a | — |

### 2.3 SLURP 意图 → FSC 动作（及默认对象）
| SLURP 意图 | action | 默认 object | 依据 |
|---|---|---|---|
| iot_hue_lighton / lightoff | activate / deactivate | lights | a |
| iot_hue_lightup / lightdim | increase / decrease | lights | a |
| iot_hue_lightchange | 不可映射 | lights | a（改颜色，FSC 没有对应动作） |
| iot_wemo_on / iot_wemo_off | activate / deactivate | **无** | a；按决定只映射动作 |
| play_music | activate | music | a + b（含 play 的 train 记录 285 / 285 为 activate music） |
| play_radio / play_podcasts / play_audiobook | activate | 无 | a |
| audio_volume_up / audio_volume_down | increase / decrease | volume | a |
| **audio_volume_mute** | **decrease** | volume | **b：含 mute 的 train 记录 99 条，动作 99 / 99 为 decrease，对象 99 / 99 为 volume** |
| audio_volume_other、music_settings | 不可映射 | volume / music | a |
| 其余 44 个意图 | 不可映射 | 无 | a |

### 2.4 由对象决定动作（覆盖 §2.3 的结果）
- object ∈ {juice, newspaper, socks, shoes} → **bring**。依据 a：31 个三元组中这些对象只与 bring 同现。
- object ∈ {Chinese, English, German, Korean} → **change language**。依据 a，同上。
- 任一实体值含整词 language（2,298 / 2,298）或 languages（92 / 92）→ **change language**，此时若没有语言名，object = `none`。依据 b。
- **不可映射**的动作计为错误，保留在分母中。

### 2.5 指标
- **预测三元组** = (action, object, location)；`none` 也作为值比较。
- **target-any（主指标）**：金标中非 `none` 的槽位里，至少一个槽位的预测值相等。
- **intent_exact**（描述）：三个槽位全部相等。
- **intent_family**（描述）：action 和 object 相等，忽略 location。
- **逐槽位准确率**、不可映射动作的比例、object 为 `none` 的比例、解析失败数：全部分路线、分条件报告。
- **双人条件**：binding-score-v1 的 `judge` 作用于 (槽位=值) 多重集，D_T = T − I，D_I = I − T；报告只命中目标、只命中干扰者、配对翻转。

## 3. 主要比较与统计（与 SLURP test 相同）
- **方向**：音频 LoRA − Whisper + T2-LoRA，在相同音频上配对。
- **三项主要比较**：clean（对应 C1）、reverb（C1-R）、spatial 选中路（C5，等量口径）。
- **Bonferroni**：每项 98.33% 区间下界 > 0 才判为显著；另报 95% 区间。
- **主指标**：target-any（§2.5），与 SLURP 一致。intent_exact 和 intent_family 只作描述（§0-3）。
- **bootstrap**：B = 2000，`default_rng(2026)`。
  - 单人条件按文本成组；
  - 空间条件按组重抽样、按夹角层分层，两次交换一起抽。
- **只作描述**：
  - 两种 LoRA 相对各自基线（原始 Qwen + P2、原始 Whisper + T2）的提升；
  - c6-real 主条件（8 扇区中心、σ10，拒答 A τ = 17°、拒答 B），自然加权用 cv_dev93 的分布；
  - 干扰者风险与按路由拆开的诊断（同 SLURP test 报告第 6 节）；
  - 文本重叠的敏感性分析；
  - pilot 64 条的探索性重算。
- 结果出来后不调整任何规则、映射或 checkpoint。失败重跑规则同 SLURP §0-8。

## 4. 条件与路线
| 条件 | 输入 | 路线 | 样本 |
|---|---|---|---|
| clean | FSC 原始 16 kHz（与 SLURP C1 链路相同，不调电平） | 原始 Qwen + P2；音频 LoRA + P2；原始 Whisper + T2；Whisper + T2-LoRA | 1,240 |
| reverb | `reverb_single.reverb()` 链路，test_eval92 RIR，每条一个不重复样例 | 同上 | 1,240 |
| spatial | 6 通道混合 → SpatialNet ep100 → v3 选路 → chain8（极性只乘一次） | 同上；选中路和另一路都推理 | 720 个样本（360 组） |
| c6-real 主条件 | 复用 spatial 两路输出，CPU 上换询问方向和选路 | 同上 | 同 spatial |
- 原始 Whisper + T2 是 pilot 没有的路线，加入是为了与 SLURP 报告的"两种 LoRA 相对各自基线"一致。
- 预处理一律使用 pilot 已核对过的 `scripts/fsc/pilot_audio.prepare()` 链路。
- C0（FSC 标准转写 → T2 / T2-LoRA）只作描述，约 1.1 GPU h（§0-4）。

## 5. 作业配置与 GPU 估计
- **GPU 作业**：<SLURM_ACCOUNT> / <GPU_PARTITION>，1 × A100，4 CPU，48 GB，20 分钟，每片 ≤ 约 14 分钟。
  - 原始 Qwen 路线用 `envs/qwen_omni`，LoRA 路线用 `envs/qwen_omni_lora`；
  - SpatialNet 用 32 GB。
- **CPU 作业**：<CPU_PARTITION>，4 CPU，16 GB，20 分钟，用于构造、选路和评分。
- 提交前工作区必须干净；每个作业记录 commit、输入和输出 sha256、模型和适配器 sha256。
- **每条耗时**（FSC pilot 实测，FSC 语音较短）：原始 P2 1.32 s，音频 LoRA 2.23 s，Whisper + T2-LoRA 2.25 s（含 Whisper）；原始 Whisper + T2 估计 1.4 s。四条路线合计约 7.2 s；每个作业加载约 1.6 分钟。

| 部分 | 推理条数 | 估计 GPU 时间 |
|---|---|---|
| clean（1,240 × 4 路线） | 4,960 | 约 2.5 h |
| reverb（1,240 × 4 路线） | 4,960 | 约 2.5 h |
| spatial（720 × 2 路 × 4 路线） | 5,760 | 约 2.9 h |
| SpatialNet 前端（720） | 720 | < 0.1 h |
| 加载开销（约 42 个作业） | — | 约 1.1 h |
| **合计** | — | **约 9.1 GPU h** |
| 可选 C0（1,240 × 2） | 2,480 | 约 1.1 h |
| 若单人条件用全部 3,054 条 | — | 合计约 16 h |

- 预算单列 **12 GPU 小时**（含 C0 和原始 Whisper + T2）。执行前按实测重估：超过 12 h 就先去掉原始 Whisper + T2 的单人条件并写明；仍超过就停下报告。

## 6. 执行顺序（已批准）
- 代码：抽样和配对 `scripts/fsc/ext_build_list.py`，混响 `ext_make_reverb.py`，任务清单 `ext_tasks.py`，推理 `ext_batch.py`，评分 `score_extension_v1.py`。
- 混合、直达声参考、SpatialNet 分离和选路直接复用 SLURP test 已核对过的脚本：`split_cfg.py` 新增 `fsc_ext` 分支，默认 devel 和 test 分支不变；作业包装脚本用 EVAL_SPLIT=fsc_ext。
1. **CPU，写代码并冻结（本次提交）**：
   - 抽样清单、配对整数规划、映射表和评分脚本；
   - devel 等价核对，用 SLURP 已有代码时要求结果逐字节一致；
   - 映射规则在 pilot 64 条上的单元测试；
   - 提交、push；之后不再改代码。
2. **CPU 构造**：单人 1,240 条、test_eval92 混响、360 组混合、直达声参考，记录全部 sha256。
3. **GPU**：SpatialNet → CPU 选路 → 推理（clean、reverb、spatial 两路）。
4. **CPU 评分**：一次性报告；另附 pilot 64 条的探索性重算。
