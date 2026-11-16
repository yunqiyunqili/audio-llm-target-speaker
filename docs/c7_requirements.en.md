English translation of c7_requirements.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c7_requirements.md

# Information needed for C7 (2026-11-05; information only, **not run, not downloaded**)

Definition of C7 (`docs/conditions.md`): speech selected by C5 or C6 → ASR → text → text LLM.

## Resources already available locally (actually checked 2026-11-05)
- Under `<SHARED_ROOT>/checkpoints/hf/hub/` there is **only** `models--Qwen--Qwen2.5-Omni-7B` (21 GB, commit ae9e1690).
- A search of the shared disk for `*whisper*` (within depth 4) found nothing. The user cache `~/.cache/huggingface/hub` is empty.
- Environment `envs/qwen_omni`: transformers 4.52.4; it has been confirmed that `WhisperForConditionalGeneration` can be imported, so running Whisper with transformers **needs no new package**; the `openai-whisper` package is not installed.
- Free space on the shared disk: 203 GB (`df`, 2026-11-05).

## ASR candidates
| Option | Model | Available? | Target path | Size | Notes |
|---|---|---|---|---|---|
| A | Qwen2.5-Omni-7B itself as ASR, with `qwen_omni_single.PROMPT` "Transcribe the speech in this audio exactly. Output only the transcription." | **available** | existing snapshot | 0 (no download) | already run successfully on clean speech in job 22003339. Drawback: the ASR and the downstream LLM are the same model, not an independent ASR |
| B | `openai/whisper-large-v3` | no | `$HF_HOME = <SHARED_ROOT>/checkpoints/hf` | about 1.55B parameters; weights about 3 GB (fp16, **not verified, not downloaded**) | 16 kHz only; loaded with transformers |
| C | `openai/whisper-small.en` or `medium.en` | no | same as above | about 0.24B / 0.77B parameters; weights about 0.5 / 1.5 GB (**not verified**) | smaller, as a control |

Before downloading, you need to confirm the target path, disk usage and job configuration. The commit also needs to be pinned.

## Text LLM plan
- Recommended: **the Qwen2.5-Omni-7B thinker with text-only input**, with the frozen **T2 prompt** (`qwen_omni_c1_p2t2.PROMPT_T2`). The T2 of job 22011390 used exactly this setup; no download needed. Text conditions use T2 as specified.
- Measured GPU memory (T2, job 22011390): inference peak 16.85 GiB, loading peak 20.83 GiB, process memory about 26.7 GB.

## Expected usage (estimates; measurements take precedence)
- Option A + text LLM: load Qwen once in one job, first obtain the text with the transcription prompt, then process the text with T2. GPU memory about 21 GiB; both steps for 200 samples estimated at 8–10 minutes in total (reference: C4/C5 take about 4 minutes per 200 samples). Configuration: <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes.
- Option B or C + text LLM: Whisper large-v3 fp16 inference is estimated at 4–6 GB of GPU memory, which fits on the same A100 40GB together with Qwen. It can also be split into two jobs: first run ASR alone and write out the text; then run T2 alone. Download about 3 GB.
- Input audio: the stream selected by C5, the same as C5: 16 kHz, −30 dBFS. The estimated-azimuth method of C6 has not yet been defined.

## Undecided items
- Which ASR to use (A, B or C); whether C7 follows C5 (oracle azimuth) or C6 (not yet defined); whether the WER of the ASR itself needs to be reported separately.
