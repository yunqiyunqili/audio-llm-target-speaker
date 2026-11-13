# C0-428 中 date 召回率在两半之间的差异（c0-date-halves-diag-v1；2026-11-06；只读，只作描述，不改划分）

- 输入：C0-428 原始输出 `outputs/qwen_omni_c0_devel428/22023517/C0/raw.jsonl`；划分按 `docs/lora_slu_plan_v2.md` §3.2（382 个 devel slurp_id 用 `random.Random(2026)` 打乱，前 191 个为选择半）。分类用 `eval_c0_devel428.classify`（c1-error-taxonomy-v1 规则，不改）。
- 脚本 `scripts/qwen/c0_date_halves_diag.py`（commit `c8adc2a`）；srun 22023770（<CPU_PARTITION>，账户 <SLURM_ACCOUNT>，4 CPU，16 GB），节点 <NODE>。结果：`outputs/c0_date_halves_diag_v1/`（不入库）。

## 数目
| | 选择半 | 报告半 | 全部 |
|---|---|---|---|
| date 金标实体 | 63 | 65 | 128 |
| 含 date 的句子 / 录音 | 53 / 62 | 50 / 64 | 103 / 126 |
| 召回率（按录音） | **0.365**（23 / 63） | **0.631**（41 / 65） | 0.500 |
| 召回率（每个句子只算一次） | 0.407（54 个） | 0.627（51 个） | 0.514 |
| 未命中分类：c 边界 / d 类型 / e 漏抽 | **11** / 6 / 23 | **1** / 4 / 19 | 12 / 10 / 42 |
| 单词 date 占比；单词 / 多词的召回率 | 0.65；0.41 / 0.27 | 0.62；0.58 / 0.72 | 0.63；0.49 / 0.51 |
- 去掉重复录音后差距仍在（0.407 vs 0.627），所以不是少数重复句子造成的。

## 最常见的 date 值（数目 / 命中）
- 选择半：today 15 / 6，tomorrow 7 / 3，**today 's 7 / 0**，this week 5 / 2，wednesday 4 / 3，**tuesday 3 / 0**，next week 3 / 3，**fifteenth 3 / 0**，thursday 3 / 1，february 2 / 2。
- 报告半：today 25 / 12，tomorrow 9 / 5，this week 6 / 5，next week 4 / 4，friday 3 / 3，today 's 2 / 0，twenty second 2 / 2，this weekend 2 / 0。

## 按意图（数目 / 命中）
- 选择半：weather_query 12 / 3，**datetime_query 12 / 4**，calendar_set 11 / 4，calendar_query 11 / 9，calendar_remove 5 / 1，news_query 3 / 0。
- 报告半：**calendar_query 17 / 16**，weather_query 15 / 5，**calendar_set 8 / 8**，general_quirky 6 / 1，datetime_query 5 / 4，news_query 5 / 4。

## 漏抽的例子（C0 输入是标准转写）
- 选择半：
  - e：`set a birthday event for wednesday`（只抽了 event_name）；`will it be cold tomorrow`；`give me the information of today's cricket match`；`what was the weather report for today`；`will it rain at all this week`；`what were the winning powerball numbers last night`；`i want to hear yesterday's podcast from my favorite channel`。
  - c（边界）：`... next tuesday at ten pm`（预测 `next tuesday`，金标 `tuesday`）；`... for the fifteenth`（预测 `the fifteenth`）；`what is today's date`（预测 `today's date`，金标 `today 's`）。
  - d（类型）：`... at one pm today`（预测 `time: one pm today`）；`... on thursday afternoon`（预测 `datetime: thursday afternoon`）。
- 报告半（几乎全是 e）：`is it going to rain tomorrow`、`what is the weather today`、`is the meeting scheduled for tomorrow`、`did i get any emails from sam today`、`i went skiing today`、`will i need sunblock this weekend`。

## 可能的原因（只是描述，没有检验）
1. **意图构成不同**：报告半的 date 集中在 calendar_query、calendar_set（25 个中命中 24 个），这两类意图里模型几乎总会抽出日期；选择半的 date 更多出现在 datetime_query、weather_query、news_query（27 个中命中 7 个），这些意图里日期常被漏掉或并进别的实体。
2. **边界不一致集中在选择半**：c 类 11 个对 1 个，主要是 SLURP 金标比模型输出更短（`tuesday` vs `next tuesday`、`fifteenth` vs `the fifteenth`、`today 's` vs `today's date`）。选择半里 `today 's` 有 7 个、全部未命中，报告半只有 2 个。
3. **样本量小**：每半只有约 50 个句子含 date，单个句子模板（如 "what is today's date" 的多条录音）就能改变几个百分点。
- 以上只描述了两半的差别，划分不变。用选择半做选择时，date 召回率的基线就比报告半低，两半的 date 召回率不宜直接比较。
