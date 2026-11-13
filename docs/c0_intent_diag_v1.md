# C0 与 C1/P2 的意图差异（c0-intent-diag-v1；2026-11-06；只读已有输出，只作描述）

> 只读两份已有输出，不推理、不改 T2 或 P2、不改评分规则。只描述，不下结论。

## 输入与方法
- C0：`outputs/qwen_omni_c0/22023448/C0/raw.jsonl`（标准转写 + T2）；C1/P2：`outputs/qwen_omni_c3_p2/22014089/C1/raw.jsonl`（干净录音 + P2）。同一批 124 条录音，金标意图和句子逐条核对一致。
- 意图是否正确：slu-score-v2 的 `score_item` 严格意图（与 C0 / C1 报告中的意图准确率相同）。
- 脚本：`scripts/qwen/c0_intent_diag.py`（commit `0dddea4`）。srun 22023498（<CPU_PARTITION>，账户 <SLURM_ACCOUNT>，4 CPU，16 GB），节点 <NODE>。结果：`outputs/c0_intent_diag_v1/c0_intent_diag.json`（不入库）。

## 计数
| | 数目 |
|---|---|
| 两者都对 | 88 |
| **C0 错、P2 对** | **7** |
| **P2 错、C0 对** | **3** |
| 两者都错 | 26 |
- 意图准确率 C0 0.734（91 / 124），P2 0.766（95 / 124），差 4 条，正好等于 7 − 3。

## C0 错、P2 对（7 条，全部）
| 录音 | 金标意图 | C0 预测 | 标准转写（C0 的输入） | P2 自己的转写 |
|---|---|---|---|---|
| audio-1490198212 | qa_factoid | general_query | where is george w bush from | Where is George W. Bush from? |
| audio-1490199179 | email_sendemail | email_query | reply to mark's email | reply to mark's email |
| audio-1490261527 | qa_factoid | datetime_query | what year was bill pullman born | What year was Bill Pullman born? |
| audio-1490366027 | qa_definition | cooking_query | what is a hibachi grill | What is a hibachi grill? |
| audio-1490368934 | iot_wemo_on | general_greet | on mobile charger plug | on off my bell charger plug |
| audio-1497270966 | qa_factoid | general_query | how big is japan | How big is Japan? |
| audio-1499244634 | play_music | music_query | play the latest pop song | Play the latest pop song |

## P2 错、C0 对（3 条，全部）
| 录音 | 金标意图 | P2 预测 | 标准转写 | P2 自己的转写 |
|---|---|---|---|---|
| audio-1498572825 | weather_query | general_greet | do i need to bring a coat today | I need to bring a coat today. |
| audio-1499244312 | calendar_query | alarm_query | what are my reminders today | What are my reminders today? |
| audio-1501694638 | qa_stock | general_quirky | what price google up to | What price Google up to |

## 分布与特征（只描述）
- **场景**：C0 错、P2 对的 7 条中，金标为 `qa` 的 4 条（qa_factoid 3、qa_definition 1），email、iot、play 各 1 条。C0 在这 4 条 qa 上预测的是 general_query（2 条）、datetime_query、cooking_query。7 条中只有 1 条（email_sendemail → email_query）场景是对的。P2 错、C0 对的 3 条场景各不相同（weather、calendar、qa），P2 在这 3 条上场景都不对。
- **两者都错的 26 条**：play 12、qa 9、email 2、calendar / takeaway / cooking 各 1。
- **句长**：C0 错、P2 对的 7 条平均 4.9 个词，没有 ≤ 2 个词的；P2 错、C0 对的 3 条平均 6.0 个词；全部 124 条平均 6.0 个词，≤ 2 个词的共 4 条。"indie" 这一条（1 个词，C0 解析失败）两者意图都错，不在这 10 条里。
- **标点和大小写**：标准转写全部小写、无标点（SLURP 原文）。P2 自己的转写中，124 条里有 109 条含标点或大写。上面 C0 错、P2 对的 7 条中，有 4 条 P2 转写里人名、地名是大写（George W. Bush、Bill Pullman、Japan）或带问号；这只是同时出现，没有检验是否是原因。
- **其他不同**：C0 和 P2 的差别不只在输入是文字还是音频，prompt 也不同（T2 与 P2 只在第一句不同，见 `qwen_omni_c1_p2t2.py`）。此外有 1 条标准转写本身与录音内容不一致（audio-1490368934：标准转写 "on mobile charger plug"，P2 听成 "on off my bell charger plug"）。
- 差 4 条（3.2 个百分点）的样本量很小，没有做显著性检验。

## 记录（2026-11-06，项目负责人决定）
- C0 与 P2 的意图准确率差异（4 条）记为**"可能是小样本波动"**，不改 T2，不再追查。
