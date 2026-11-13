# C1 语义测试方案（Qwen2.5-Omni-7B，结构化意图 + 实体；已运行：作业 22010083）

## 目的
在干净的近讲语音上（C1，原始 16 kHz），确认 Qwen 能按指定格式输出意图和实体，并让解析和规范化规则跑通。只用 20 条，**不代表整体性能**，也不用 test。

## 样本（已生成清单，未生成任何音频）
- 脚本 `scripts/slurp/select_c1_semantic_devel.py`，srun（<CPU_PARTITION> / 1 CPU / 4 GB），清单在 `configs/c1_semantic_devel20.json`。
- 候选范围：口径 A，官方 devel 句子 × devel 说话人（test 40% / devel 20% 的说话人方案，seed 2026），并去掉与 test 文本相同的句子。共 146 条录音、10 个说话人。不管最后从 train 留出多少比例，这部分都属于 devel，所以这 20 条不受比例选择的影响。
- 抽样：候选按文件名排序后，用 `random.Random(2026)` 打乱，再按"每个意图最多几条、每个说话人最多几条"分轮贪心选取，限制依次为 (1,1)、(1,2)、(2,2)、不限，直到选满 20 条，slurp_id 不重复。结果：20 条，覆盖 18 个意图、10 个说话人，时长 1.6–7.5 s，每条都至少有 1 个实体。
- 意图标签用 `scenario_action`。原因是官方 jsonl 的 `intent` 字段有 229 个句子与 scenario_action 不一致，例如 `query`、`music`、`set` 这类缺了场景前缀；按 scenario_action 计，官方 train 共 60 个意图。

## Prompt
- system：Qwen 默认的 system prompt，与单条推理相同。
- user：一段音频，加上下面的文本。标签清单来自官方 train：60 个意图、55 种实体类型，存放在 `configs/slurp_label_inventory_train.json`；整段文本共 2,043 个字符。
```
This audio is a spoken command to a voice assistant. Return ONLY one JSON object, with no other text, in exactly this format:
{"transcript": "<the words spoken>", "intent": "<one intent label>", "entities": [{"type": "<one entity type>", "value": "<the entity words as spoken>"}]}
Use "entities": [] if there are none.
Allowed intent labels: <60 个 scenario_action，逗号分隔>
Allowed entity types: <55 个类型，逗号分隔>
```
- 解码：贪心（do_sample=False），max_new_tokens=256，batch=1，只输出文本（talker 关闭）。

## 输出格式与解析规则（`scripts/qwen/slu_normalize.py: parse`）
1. 去掉 ```json 代码块标记；取第一个 `{` 到最后一个 `}` 之间的内容，用 `json.loads` 解析。
2. 状态分四种：`ok`、`no_json`（找不到 JSON）、`bad_json`（无法解析）、`bad_schema`（不是对象，或 entities 不是列表）。除 `ok` 外都计为意图错误、实体为空，并分别计数。
3. 字段：`transcript`、`intent` 不是字符串时记为空串；`entities` 中只保留 type 和 value 都是字符串的项，丢弃的项单独计数；type 统一转成小写。
4. 输出的意图标签如果不在 60 个标签之内，计为错误，并单独计数（`intent_invalid_label`），不做近似匹配。

## 规范化规则（`scripts/qwen/slu_normalize.py: normalize`；对参考和输出同样处理）
1. NFKC，转小写，弯引号转成直引号；`a.m.`/`p.m.` 转成 `am`/`pm`。
2. `%` 转成 " percent"，`&` 转成 " and "，连字符和斜杠转成空格。
3. 时间 H:MM 转成英文单词：`5:30` → five thirty，`5:00` → five，`5:05` → five oh five。
4. 序数：`21st` → twenty first。0–9999 的整数转成基数词：`5` → five。小数：`2.5` → two point five。
5. 合并撇号分开的词：`derek 's` → derek's（SLURP 的实体值由 token 拼接而成，所以会出现这种写法）；除撇号外的标点全部删除；合并多余空格。
- 已知限制，未处理：
  - 年份按基数词读，`1990` → one thousand nine hundred ninety，与口语 "nineteen ninety" 不一致。
  - `R&B` → "r and b"，与 SLURP 参考值 "r n b" 不一致。
  - "grass market" 与 "grassmarket" 只在"忽略空格"的次要指标里算对。
- 自检：`python -I scripts/qwen/slu_normalize.py` 的 11 个固定用例全部 OK（2026-11-04 在登录节点运行，不涉及数据）。

## 指标（`scripts/qwen/qwen_omni_c1_semantic.py`）
- 意图准确率：与 scenario_action 完全一致才算对。
- 实体：(type, 规范化后的 value) 按多重集合匹配，计算 P、R、F1；另报告一个忽略 value 内部空格的 F1。
- 转写 WER：参考句和输出 transcript 都规范化后计算编辑距离。
- 另外记录：各解析状态的计数、超出标签清单的意图数、每条的推理时间和生成 token 数、加载时间、峰值显存。每条的原始输出完整保存在 `raw.jsonl`。
- **未实现**：SLURP 官方的 SLU-F1（基于词级和字符级编辑距离的实体软匹配）。

## 作业配置与预计占用（提交前写的，实测见文末）
- 脚本 `scripts/qwen/qwen_omni_c1_semantic.sbatch` 调用 `scripts/qwen/qwen_omni_c1_semantic.py`。`sbatch --test-only` 通过，预计 2026-11-05 00:22 开始，节点 <NODE>。
- 配置与单条推理相同：<SLURM_ACCOUNT> / <GPU_PARTITION> / 1×A100 / 4 CPU / 48 GB / 30 min。离线模式，从本地 snapshot 目录（commit ae9e1690）加载。
- 预计占用，依据作业 22003339 的实测值：
  - 加载约 47 s，峰值显存约 20.8 GiB，进程内存约 26 GB。
  - prompt 比单条推理长约 2 KB 字符、输出最多 256 token。单条推理时第二条用了 0.33 s 生成 11 个 token；这次每条可能要几秒到十几秒，20 条估计不超过 5 分钟。这只是估计，以实测为准。
  - 输出文件 `outputs/qwen_omni_c1_semantic/<jobid>/` 中的 raw.jsonl 和 summary.json，估计小于 100 KB；不写音频。

## 实测结果（作业 22010083；只有 20 条，不代表整体性能，未据此修改 prompt）
- 运行：<NODE>，2026-11-05 00:45:54–00:48:50（2 min 56 s），退出码 0，MaxRSS 26.4 GB。加载 56.0 s，加载峰值显存 20.83 GiB，推理峰值 16.93 GiB；20 条推理共 27.5 s（每条 0.74–2.41 s）。
- 输出：`outputs/qwen_omni_c1_semantic/22010083/{raw.jsonl,summary.json}`；离线重评分（只读 raw.jsonl，登录节点运行，不用 GPU）：`rescore_summary.json`、`rescore_errors.jsonl`。
- 格式：20/20 解析为 ok；丢弃的实体 0；意图超出清单 0；达到 max_new_tokens 0；实体类型超出清单 1（`location`）。

| 指标 | 严格 | 放宽 |
|---|---|---|
| 意图 | 14/20 = 0.70 | 只看场景前缀：16/20 = 0.80 |
| 实体 F1（hyp 22，ref 30） | TP 10，P 0.455，R 0.333，F1 0.385 | 只比 value：TP 15，F1 0.577；同类型且有共同词：TP 12，F1 0.462；忽略空格：TP 11，F1 0.423 |
| 转写 WER（规范化后） | 8/154 = 0.052 | — |

- 有错误的样本 17/20。自动错误类型（按样本计，一条可多类）：entity_missing 13、intent 6、transcript 5、entity_spurious 5、entity_type 5、entity_boundary 1、parse 0。每条的参考句、参考标签、原始输出和错误类型见 `rescore_errors.jsonl`。

## C1 扩展：P0 / P1 / T0（作业 22010997；100 条 devel；评分 slu-score-v2）
- 样本 `configs/c1_ext_devel100.json`；脚本 `scripts/qwen/qwen_omni_c1_ext.py` + `.sbatch`；<NODE>，2026-11-05 09:54:24–10:01:46（7 min 22 s），退出码 0，MaxRSS 26.5 GB；加载 49.4 s，加载峰值 20.83 GiB。
- 输出 `outputs/qwen_omni_c1_ext/22010997/{P0,P1,T0}/`（raw.jsonl 含每条完整 prompt；rescore_v2_*）、`condition_diff.jsonl`、`condition_diff_summary.json`。

| | P0 音频 | P1 音频+3 例 | T0 文本 |
|---|---|---|---|
| 解析 ok | 100 | 99（1 条 no_json） | 99（1 条 no_json） |
| 意图严格 / 场景前缀 | 77 / 81 | 72 / 76 | 71 / 76 |
| 实体严格 F1（TP，hyp/ref=128） | 0.362（36，71） | 0.333（32，64） | 0.398（40，73） |
| 只比值 F1 / 词重叠 F1 | 0.452 / 0.452 | 0.458 / 0.427 | 0.527 / 0.448 |
| 严格完整语义正确 | 19 | 16 | 17 |
| WER（v2 / v1 规范化） | 36/622=0.058 / 0.058 | 38/622=0.061 / 0.061 | — |
| 意图不在清单 / 实体类型不在清单 | 3 / 4 | 1 / 5 | 5 / 5 |
| 推理总时间 / 推理峰值显存 | 106.1 s / 16.93 GiB | 102.6 s / 16.98 GiB | 101.3 s / 16.84 GiB |

只有 100 条 devel，单次运行，不代表整体性能；没有据此选择 prompt。
