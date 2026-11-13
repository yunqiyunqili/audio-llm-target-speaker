# C7 方案 A v1：Qwen 先转写，再用 T2 抽取（c7-planA-v1；2026-11-05；在运行前写定）

- 输入：C5 的目标路，与 C5 送给 Qwen 的音频完全相同：FaSNet-TAC 的 v3 选路，加极性修正 → `resample_poly(x, 2, 1)[:2T]` → −30 dBFS。**依赖 oracle 方位**，与 C5 相同。
- 第一步，转写：用 Qwen2.5-Omni-7B（commit ae9e1690，同一个本地 snapshot），system prompt 是默认的 SYSTEM，user 内容为 [音频, 文本]，文本是：
  > Transcribe the speech in this audio exactly. Output only the transcription.
  - 这句就是 `scripts/qwen/qwen_omni_single.py` 中的 `PROMPT`，在作业 22003339 的干净语音上用过。**只跑这一版，不调 prompt。**
  - 解码：贪心，max_new_tokens 256，batch 1，只输出文本。
- 第二步，抽取：把第一步的原始转写文本去掉首尾空白，加一个换行后作为文本输入，再接冻结的 **T2 prompt**（`qwen_omni_c1_p2t2.PROMPT_T2`），不给音频。user 内容为 [转写 + 换行, PROMPT_T2]，格式与 T2 相同。解码设置同上。
- 不运行 Whisper，不下载任何模型。
- 评估：
  - 用 slu-score-v2 和 binding-score-v1（`eval_c4_v1.score` 和 `summ`），指标与 C5 相同：四类比例、目标至少命中一个、目标全部命中、配对翻转准确率、M、U、解析失败、拒绝执行。
  - 另外报告第一步转写与目标句的 WER（normalize_v2），以及与干扰句的 WER。
  - 与 C5、C4-direct 比较。只作描述，不设判定。
- 实现：`scripts/qwen/qwen_omni_c6c7.py`（与 C6δ 的另一路放在同一个作业）；评估 `scripts/qwen/eval_c7_planA.py`。

## 结果（2026-11-05；**依赖 oracle 方位**，与 C5 相同）
- 运行记录：作业 22015445（与 C6δ 同一个作业），转写 C7ASR 用时 185.9 s，抽取 C7T2 用时 329.3 s。评估：`scripts/qwen/eval_c7_planA.py`，结果在 `outputs/c7_planA_v1/eval_c7_planA.json`。

| 条件 | 只含目标 | 只含干扰 | 两者都不含 | 目标至少命中一个 | 目标全部命中 | 配对翻转准确率 | M | U | 解析失败 / 拒绝执行 |
|---|---|---|---|---|---|---|---|---|---|
| **C7 方案 A** | 0.215 | 0.005 | 0.780 | **0.255** | 0.215 | 0.06 | 33 | 139 | **11 / 5** |
| C5 | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 | 82 | 0 / 0 |
| C4-direct | 0.565 | 0 | 0.435 | 0.670 | 0.565 | 0.36 | 185 | 15 | 0 / 0 |

- 第一步的转写质量：
  - 与目标句的 WER 为 3.96，与干扰句为 4.28；
  - 只有 30/200 与目标句的 WER < 0.5；
  - 122/200 的输出长度超过参考句的 2 倍；
  - 60/200 含有 "sorry"、"can't"、"I don't know" 一类字样；
  - 有 2 条含非英文字符。
- 例子：
  - g001_s1 的转写是 "I don't know what your favorite song by Bruno Mars is. You should tell me so we can talk about it."；
  - g001_s2 的转写是 "I'm sorry, I can't hear the audio. …"。
- 描述：用这一版转写 prompt 时，Qwen 在分离后的语音上**经常不转写，而是直接回应指令内容**。所以 C7 方案 A 比 C5 低 0.14。按约定只跑了这一版 prompt，没有调整。不运行 Whisper，不下载模型。
