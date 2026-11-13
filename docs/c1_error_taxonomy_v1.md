# C1 错误分类 v1（c1-error-taxonomy-v1；2026-11-06；规则在运行前写定，看到结果后不修改）

> - 目的：判断 C1（干净近讲录音）上 P2 没有命中目标实体，主要是"写法不同"还是"真没抽对"。
> - **只读已有输出，CPU 运行**。不推理、不训练、不下载、不改环境、不动训练作业、不碰 test。不修改 mix_v1、P2/T2 prompt、slu-score-v2、binding-score-v1、v3 选路规则、已有结果和判定。
> - 本文档里的规范化（"诊断规范化"）**只用于本诊断**，不替代 normalize_v2 或 binding-score-v1。

## 1. 输入
| 路线 | 文件 | 来源 |
|---|---|---|
| C1 / P2 | `outputs/qwen_omni_c3_p2/22014089/C1/raw.jsonl` | 作业 22014089（`docs/c3_results_v1.md`，"C1 / P2 0.694"；该值存于 `outputs/qwen_omni_c3_p2_eval/22014089/eval_c3_summary.json` 的 `C1.mean_target_any`） |
| C7W-C1：T2 输出 | `outputs/qwen_omni_c7w_f1/22016221/T_C1/raw.jsonl` | 作业 22016221（`docs/c7_whisper_v1.md`） |
| C7W-C1：Whisper 转写 | `outputs/qwen_omni_c7w_f1/22016221/W_C1/raw.jsonl` | 同上 |
- 金标实体：`configs/mix_v1_devel100_v2.json` 中各组 A、B 录音的 `entities`（即 binding-score-v1 所用的 SLURP 金标）。
- 检查：三个文件都必须存在，各自恰好覆盖 mix_v1 的 124 条录音（每条一行）；`T_C1` 的 `asr_text` 必须与 `W_C1` 的 `raw_output` 一致。**任何一项不满足就停下报告，不换数据。**

## 2. 单位与"命中"
- 单位：每条录音的每个金标实体（类型, 值）一条。
- **命中**（与 binding-score-v1 相同的匹配方式，但不限于区分值）：预测实体值经 normalize_v2 后的多重集合 H（不看类型）。按金标实体的顺序依次检查：normalize_v2(金标值) 在 H 中尚有剩余计数则为命中，并消耗一个计数。
- 只对**未命中**的金标实体分类。分母同时报告：金标实体总数、有实体的录音数、录音总数；并另报告"在至少一组配对中属于目标区分值（D_T）的实体"这一子集的分类结果。

## 3. 诊断规范化 N(s)
依次：
1. 小写；`a.m.`/`a. m.` → `am`，`p.m.` → `pm`；
2. 时间：`H:00` → `H`；`H:MM` → `H MM`（H 为 1–2 位数字）；`o'clock` / `oclock` 删除；
3. 标点（除字母、数字、空格以外的字符）替换为空格，合并空格，去首尾空格；
4. 英文数词 → 阿拉伯数字：zero–nineteen、twenty–ninety，以及"十位 + 个位"的组合（如 `twenty five` → `25`）；其余词不变。序数词不转换。
- **值相同**：N(a) = N(b)；或者两者中**只有一方**含 `am`/`pm` 词、**另一方原文是钟点写法**（含 `H:MM`），删去 `am`/`pm` 后相同。这样 `7:00`、`7 am`、`seven am` 互相相同；`7 am` 与 `7 pm` 不同；`seven` 与 `seven am` 不算相同（归 c 类边界不同，与任务说明中的例子一致）。
- （修订说明：本条在运行前、首次提交 `9cc3ef9` 之后修订过一次，原写法会把 `seven` 与 `seven am` 判为相同，与任务说明的例子矛盾；修订后才运行。）
- **词重叠**：N(a) 与 N(b) 的词集合（删去 `am`/`pm` 之前）交集非空。不设停用词；另报告 c、d 类中"重叠的词全是停用词"（the、a、an、of、to、in、on、at、for、my、me、and、is、it、this、that）的数目，只作描述。

## 4. 分类（按顺序，每个未命中实体只归一类）
| 类 | 条件 |
|---|---|
| a. 解析失败 | 该条输出 `parse()` 的状态不是 `ok` |
| b. 写法不同 | 预测中有**同类型**实体，与金标值"值相同" |
| c. 边界不同 | 预测中有**同类型**实体，与金标值"词重叠"但不"值相同" |
| d. 类型错 | 预测中有**不同类型**的实体，与金标值"值相同"或"词重叠" |
| e. 漏抽 | 以上都不满足，但 N(金标值) 作为连续词序列出现在 N(transcript) 中 |
| f. 听错 | 以上都不满足，N(金标值) 不在 N(transcript) 中 |
- 类型比较：小写、去首尾空格后相等。
- 每个金标实体独立与所有预测实体比较（不做一对一分配）。
- transcript：P2 用模型自己输出 JSON 中的 `transcript`；Whisper + T2 用 **Whisper 转写**（`W_C1` 的 `raw_output`）。

## 5. 输出
- 各类的实体数和比例（占未命中实体），P2 和 Whisper + T2 并列；全部金标实体和 D_T 子集各一张表。
- 每类最多 10 个例子（按录音文件名排序取前 10）：录音 id、金标实体、预测实体、transcript。
- **把 b 类算作命中后的"目标至少命中一个"**（只作诊断，不替代 binding-score-v1）：把 b 类金标值的 normalize_v2 值加入 H，其余计算不变：
  - 按录音平均（与 `eval_c3_v1.py` 相同：每条录音按它所在各组的配对计算，再对录音平均）；
  - 映射到 200 个样本（与 `eval_c7w_f1.py` 相同：目标录音的输出 + 样本自己的配对，`eval_c4_v1.score(single=True)`）。
  - 一致性：不加 b 类时，P2 按录音的值必须等于 0.6935…（`eval_c3_summary.json`），Whisper + T2 映射到 200 个样本的值必须等于 0.650（`outputs/c7w_f1_eval/22016732_full/eval_c7w_f1.json` 的 `C7W_C1`），否则停下。
- 不设判定阈值，只作描述。

## 6. 实现与运行
- 脚本：`scripts/qwen/c1_error_taxonomy.py`；结果：`outputs/c1_error_taxonomy_v1/`（不入库）。
- 工作区干净时在 <CPU_PARTITION> 上 srun 运行：账户 <SLURM_ACCOUNT>，4 CPU，16 GB，30 分钟；环境 `envs/qwen_omni`（只用标准库和已有脚本）。

---

## 结果（2026-11-06）
- 运行记录：srun 22023424（<CPU_PARTITION>，账户 <SLURM_ACCOUNT>，4 CPU，16 GB，30 分钟），节点 <NODE>，13:32:46–13:32:49，退出码 0。运行时 HEAD 为 `88c0351`，工作区干净。结果在 `outputs/c1_error_taxonomy_v1/`（`c1_error_taxonomy.json`、`entities_P2.jsonl`、`entities_T2.jsonl`，不入库）。
- 输入检查通过：三个文件各 124 条录音，与 mix_v1 一致；T_C1 的 `asr_text` 与 W_C1 逐条相同；raw 中的 `ref_entities` 与 mix_v1 配置中的金标逐条相同。
- 一致性通过：不加 b 类时，P2 按录音的"目标至少命中一个"= 0.6935（与 `eval_c3_summary.json` 相同），Whisper + T2 映射到 200 个样本 = 0.650（与 `eval_c7w_f1.json` 相同）。

### 分母
- 124 条录音，**全部**有金标实体；金标实体共 **144** 个。
- 144 个实体**全部**在至少一组配对中属于目标区分值（D_T），所以 D_T 子集的结果与全部实体完全相同，下面只列一张表。
- 命中：P2 86 / 144（未命中 58），Whisper + T2 81 / 144（未命中 63）。

### 未命中实体的分类
| 类 | P2：数目（占未命中） | Whisper + T2：数目（占未命中） |
|---|---|---|
| a. 解析失败 | 0（0%） | 0（0%） |
| b. 写法不同 | **0（0%）** | **0（0%）** |
| c. 边界不同 | 11（19.0%） | 12（19.0%） |
| d. 类型错 | 6（10.3%） | 6（9.5%） |
| e. 漏抽（值在 transcript 里） | **28（48.3%）** | **30（47.6%）** |
| f. 听错（值不在 transcript 里） | 13（22.4%） | 15（23.8%） |
| b + c + d | 17（29.3%） | 18（28.6%） |
- c、d 类中"重叠的词全是停用词"：P2 0 个，Whisper + T2 1 个。
- 按金标类型：P2 的 58 个未命中里 `date` 有 19 个（`date` 共 43 个金标），其中 17 个是 e 类；Whisper + T2 的 63 个未命中里 `date` 有 20 个，其中 16 个是 e 类。

### 把 b 类算作命中后的"目标至少命中一个"（只作诊断）
| 路线 | binding-score-v1：按录音 / 200 个样本 | b 类算命中：按录音 / 200 个样本 |
|---|---|---|
| P2 | 0.694 / 0.695 | 0.694 / 0.695（b 类为 0，不变） |
| Whisper + T2 | 0.653 / 0.650 | 0.653 / 0.650（不变） |

### 例子（P2；每类最多 10 个，完整列表在 json 中）
- **c 边界不同**：`person "mark 's"` → 预测 `mark`；`person "barack obaba"` → `Barack Obama`；`person "micheal jackson"` → `Michael Jackson`；`game_name "temple runner"` → `Temple Run`；`list_name "school supplies"` → `school supplies list`；`music_genre "emo music"` → `emo`；`device_type "mobile charger plug"` → `bell charger plug`；`weather_descriptor "temperature"` → `highest temperature`；`weather_descriptor "sunscreen"` → `sunscreen day`；`radio_name "next howard stern show"` → `Howard Stern show`。
- **d 类型错**：`house_place "bathroom"` → `device_type "upstairs bathroom light"`；`definition_word "volkswagen golf car"` → `car_model "Volkswagen Golf"`；`podcast_descriptor "stopped last night"` → `podcast_name "that podcast I stopped last night"`；`business_type "dress shows"` → `clothing_type "dress shorts"`；`place_name "within one mile"` → `distance "one mile"`；`podcast_descriptor "next podcast episode"` → `podcast_name "next podcast"`。
- **e 漏抽**：多数是日期没有抽出，例如 "What are my reminders today?"（金标 `date today`，预测 `none none`）、"Will it be cold tomorrow?"（金标 `date tomorrow`，只预测了 `weather_descriptor cold`）、"Is it going to rain tomorrow?"、"Will I need sunblock this weekend?"、"What is the weather today?"；另有 `event_name holiday`、`definition_word "this sunset"`。
- **f 听错**：`artist_name "bob dylan"`（transcript "Play newest pop didn't playlist"）、`person "angelina jolie"`（"How old is engineer Joey?"）、`music_genre indie`（"India"）、`business_name chipotle`（"Close this trip to."）、`definition_word regal`（"rego"）。

### 观察（只描述，规则不因此修改）
1. **b 类（写法不同）为 0**：在诊断规范化下，没有一个未命中实体只是数字、时间或大小写写法不同。所以按本规则，C1 上 P2 没命中**不是**"写法不同"造成的。
2. 最大的一类是 **e 漏抽（约 48%）**：值已经在模型自己的 transcript 里，但没有作为实体输出，其中多数是 `date`（today、tomorrow、this weekend 等）。另有约 22% 是 f 听错，约 29% 是 c + d（边界或类型与 SLURP 标注不一致）。
3. P2 与 Whisper + T2 的分类比例几乎相同，差别主要是 Whisper + T2 的 e、f 各多 2 个。
4. **规则本身的局限**（看到结果后发现，规则不改，只记录）：
   - 诊断规范化把撇号替换成空格，所以 `applebees` 与 `Applebee's`（`applebee s`）不相等，被归为 f。事后按"删去撇号"重算：两条路线中各有 1 个实体（`applebees`）会变成 b 类，另有 f 类的 3 个（`applebees`、`tomorrows`、`doctors appointment`）会出现在 transcript 里。这只是事后描述，不替代上表。
   - c 类中有几个是 **SLURP 金标本身的拼写错误**（`barack obaba`、`micheal jackson`），模型的写法其实是对的。本规则无法把它们与真正的边界错误区分开。
   - 每个金标实体独立与所有预测实体比较，同一个预测实体可能同时解释多个金标实体。
