# SLU 评分规则（slu-score-v2，2026-11-05 冻结）

**冻结约定**：进入 test 之前就冻结这套规则。之后如果要改，必须升级版本号（v3），用同一套新规则重算所有 devel 结果，并写清楚改了什么。用过 test 之后不再修改。旧版本的结果文件一律不覆盖。

## 冻结的文件
| 文件 | sha256 |
|---|---|
| `scripts/qwen/slu_normalize.py`（解析和 v1 规范化） | `d623281c7bd83029197dfca3bc99927b5cac951f12562309d343a77247ed72c3` |
| `scripts/qwen/rescore_slu_v2.py`（v2 评分） | `00ad15d80a9688825988767adcbe67f246609cc0fc061a6be3f259ba0e8fa498` |
| `scripts/qwen/compare_c1_ext_conditions.py`（条件间逐条对比，不打分） | `a77b20eaffe42fc6795a959292b2c682a09e468cb7bd00564cf7cc4a76dd98c3` |
| `configs/slurp_label_inventory_train.json`（60 个意图、55 种实体类型，来自官方 train） | 未改动 |

## 输入
只读 `raw.jsonl`（模型原始输出），不重新生成。参考标签来自 SLURP commit 8eb1654；意图是 `scenario_action`，实体是 (type, value)，value 由 SLURP 标注的 token 拼接而成。

## 解析（与 v1 相同）
去掉 ``` 代码块标记，取第一个 `{` 到最后一个 `}` 之间的内容，用 `json.loads` 解析。状态分为 ok / no_json / bad_json / bad_schema；非 ok 时意图记为错，实体记为空。type 转小写；type 或 value 不是字符串的实体丢弃并计数。

## 规范化
- v1（`normalize`）：见 `docs/c1_semantic_plan.md`。
- v2（`normalize_v2`）：先做 v1，再把 `rnb`、`r'n'b`、`r and b`、`r n b` 统一为 `r n b`。
- 所有格：判断实体边界和计算词重叠时，词尾的 `'s` 去掉后再比较（"derek's" 和 "derek" 有共同词）。严格匹配**不**去掉。

## 指标
| 指标 | 定义 |
|---|---|
| 意图严格准确率 | 解析为 ok，且意图与 scenario_action 完全相同 |
| 场景前缀准确率 | 意图中第一个 `_` 之前的部分相同 |
| 实体严格 F1（v2） | (type, normalize_v2(value)) 按多重集合匹配；`strict_v1` 用 v1 规范化，保留以便和旧结果对照 |
| 只比值 F1 | 忽略 type，只用 normalize_v2(value) 按多重集合匹配。**不做**人工的相近类型映射 |
| 词重叠 F1 | type 相同、且（去掉所有格后）至少有一个共同词；按参考顺序贪心一对一匹配 |
| 严格完整语义正确率 | 意图严格正确，**且**实体多重集合与参考完全相同（v2 规范化）；不看转写。另报 v1 规范化下的值 |
| 转写 WER | normalize_v2 后计算词编辑距离；`wer_v1` 用 v1 规范化。只对音频条件报告 |
| 其他计数 | 解析状态、丢弃的实体、超出清单的实体类型和意图、达到 max_new_tokens 的条数 |

## 自动错误类型（每条可以有多个）
`parse`；`transcript`（WER>0，仅音频条件）；`intent`；实体先按严格匹配去掉配对上的，剩下的参考实体依次判为：`entity_type`（值相同、类型不同）→ `entity_boundary`（类型相同、有共同词，包括所有格差异）→ `entity_missing`；剩下的输出实体如果和剩余参考实体既不同值、也不是同类型且有共同词，判为 `entity_spurious`。"标注问题"不自动判定。

## 争议项
`configs/c1_disputed_items.json` 里列出的样本在错误明细中标为 `review_status = 待确认`，**分数不变**。按 2026-11-05 的决定不听音频，等 100 条结果出来后再抽查。

## 校验（2026-11-05，作业 22010083 的 20 条）
- `strict_v1` 与 v1 的 `rescore_summary.json` 一致：TP 10，F1 0.385；v1 文件的 md5 在重评分前后不变。
- v2：严格实体 F1 0.423（TP 11，多出的 1 个是 "RnB"），只比值 0.615，词重叠 0.538，严格完整语义 4/20（v1 规范化下 3/20），WER 0.032（v1 规范化 0.052）。
