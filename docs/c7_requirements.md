# C7 所需信息清单（2026-11-05；只列信息，**未运行、未下载**）

C7 的定义（`docs/conditions.md`）：C5 或 C6 选出的语音 → ASR → 文本 → 文本 LLM。

## 本地已有的资源（2026-11-05 实际查看）
- `<SHARED_ROOT>/checkpoints/hf/hub/` 下**只有** `models--Qwen--Qwen2.5-Omni-7B`（21 GB，commit ae9e1690）。
- 共享盘查找 `*whisper*`（深度 4 层以内），没有找到任何结果。用户缓存 `~/.cache/huggingface/hub` 是空的。
- 环境 `envs/qwen_omni`：transformers 4.52.4，已确认可以 import `WhisperForConditionalGeneration`，所以用 transformers 跑 Whisper **不需要装新包**；`openai-whisper` 包没有安装。
- 共享盘剩余空间：203 GB（`df`，2026-11-05）。

## ASR 候选
| 方案 | 模型 | 是否已有 | 目标路径 | 大小 | 说明 |
|---|---|---|---|---|---|
| A | Qwen2.5-Omni-7B 本身作 ASR，用 `qwen_omni_single.PROMPT` "Transcribe the speech in this audio exactly. Output only the transcription." | **已有** | 现有 snapshot | 0（不用下载） | 作业 22003339 中已在干净语音上跑通。缺点：ASR 和下游 LLM 是同一个模型，不是独立的 ASR |
| B | `openai/whisper-large-v3` | 没有 | `$HF_HOME = <SHARED_ROOT>/checkpoints/hf` | 约 1.55B 参数；权重约 3 GB（fp16，**没有核实，没有下载**） | 只支持 16 kHz；用 transformers 加载 |
| C | `openai/whisper-small.en` 或 `medium.en` | 没有 | 同上 | 约 0.24B / 0.77B 参数；权重约 0.5 / 1.5 GB（**没有核实**） | 更小，作对照 |

下载前需要你确认目标路径、磁盘占用和作业配置。commit 也要固定下来。

## 文本 LLM 方案
- 推荐使用 **Qwen2.5-Omni-7B 的 thinker，只输入文本**，配冻结的 **T2 prompt**（`qwen_omni_c1_p2t2.PROMPT_T2`）。作业 22011390 的 T2 用的就是这一套，不用下载。文本条件按规定就是用 T2。
- 实测显存（T2，作业 22011390）：推理峰值 16.85 GiB，加载峰值 20.83 GiB，进程内存约 26.7 GB。

## 预计占用（估计，以实测为准）
- 方案 A + 文本 LLM：一个作业里加载一次 Qwen，先用转写 prompt 得到文本，再用 T2 处理文本。显存约 21 GiB，200 个样本两步合计估计 8–10 分钟（参考 C4/C5 每 200 个样本约 4 分钟）。配置：<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、48 GB、30 分钟。
- 方案 B 或 C + 文本 LLM：Whisper large-v3 fp16 推理显存估计 4–6 GB，和 Qwen 放在同一张 A100 40GB 上可以装下。也可以拆成两个作业：先单独跑 ASR，写出文本；再单独跑 T2。下载约 3 GB。
- 输入音频：C5 选出的那一路，与 C5 相同：16 kHz，−30 dBFS。C6 的估计方位方法还没有定义。

## 尚未决定的事项
- 用哪一个 ASR（A、B 还是 C）；C7 接 C5（oracle 方位）还是 C6（还没有定义）；是否需要把 ASR 本身的 WER 单独报告出来。
