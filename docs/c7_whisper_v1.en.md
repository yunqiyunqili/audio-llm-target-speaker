English translation of c7_whisper_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c7_whisper_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C7-W v1: Whisper transcription + T2 extraction (c7-whisper-v1; 2026-11-05; rules written before the run and not changed after the results were seen)

## ASR model (to be downloaded after confirmation)
- `openai/whisper-large-v3`, revision **`06f233fe06e710322aca913c1bc4249a0d71fce1`** (the Hugging Face API shows lastModified 2024-08-12; no access request needed; licence apache-2.0; only the metadata was queried on 2026-11-05).
- Only the fp16 single-file weights and the necessary configuration files are downloaded, specified with `allow_patterns`, 11 files in total, **3,091,519,764 B (2.88 GiB)**:
  - `model.safetensors`: 3,087,130,976 B, LFS sha256 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`;
  - the other 10 files: config.json, generation_config.json, preprocessor_config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, added_tokens.json, special_tokens_map.json, normalizer.json.
  - The fp32 shards, flax and pytorch_model.bin are **not downloaded** (the whole repository is 24.7 GB).
- Target path: `HF_HOME = <SHARED_ROOT>/checkpoints/hf`, i.e. `hub/models--openai--whisper-large-v3/snapshots/06f233fe…/`. 203 GB free on the shared disk.
- Download method: the same as for Qwen, with `huggingface_hub.snapshot_download(repo_id, revision=<sha>, allow_patterns=[…])`.
  - Environment: the huggingface_hub already in the tfgridnet environment; nothing installed;
  - planned to run with srun on a CPU node: <SLURM_ACCOUNT>, <CPU_PARTITION>, 2 CPU, 4 GB, 30 minutes;
  - after downloading, check each file size against the metadata, and verify the sha256 of model.safetensors.
- Loading: from the local snapshot directory, offline (`HF_HUB_OFFLINE=1`), `local_files_only=True`, the same as for Qwen, to avoid the revision problem met before 22003339. The environment is qwen_omni (transformers 4.52.4; it has been confirmed that `WhisperForConditionalGeneration` can be imported); no new packages installed.

## Download record (2026-11-05, approved)
- srun 22016212 (<CPU_PARTITION>, huggingface_hub 0.34.4 of the tfgridnet environment), took 9.7 s, downloaded only the 11 files.
- Check results:
  - 11 files, total size 3,091,519,764 B, every file size matching the metadata;
  - the sha256 of `model.safetensors` is `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`, matching the LFS metadata (srun 22016215).
- Disk: 2.9 GB; 203 GB free on the shared disk.

## Conditions
All three groups of speech are 16 kHz, obtained as follows. Generated only in memory; no audio written:
| Group | Audio | Samples |
|---|---|---|
| C7W-C1 | SLURP original 16 kHz close-talk recordings (same as C1) | the 124 recordings in mix_v1; mapped to the 200 samples in evaluation, the same way as R1 (taking the output of the target recording and judging it with the sample's own pair) |
| C7W-C4direct | the target's direct reference → `resample_poly(x, 2, 1)` → −30 dBFS (same as C4-direct) | 200 |
| C7W-C5 | the target stream selected by C5 → polarity correction → 8→16 kHz → −30 dBFS (same as C5) | 200 |

- Whisper decoding: fp16, features extracted with `WhisperProcessor`, `generate(language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128)`, no timestamps, batch 1.
- Text extraction: the Whisper transcript after `.strip()`, followed by a newline, then the frozen **T2 prompt**, with Qwen2.5-Omni-7B on text-only input. Decoding the same as T2: greedy, max_new_tokens 256.
- C7W-C5 depends on the oracle azimuth, as C5 does.

## Metrics
- Using `eval_c4_v1.score` and `summ`, unchanged: target only, interferer only, neither, target-any, all target values hit, pair-flip accuracy, M, U, parse failures, refusals.
- Also reported: the WER of the Whisper transcript against the target sentence and against the interferer sentence (normalize_v2).

## Verdict (empirical, not significance tests; a = target-any over 200 samples; existing results not changed)
- **Confirmed (2026-11-05)**: the verdict compares C7W-C5 with C7W-C4direct; C4-direct under P2 is reference only. Both go through the same transcription-plus-text chain:
  - |a(C7W-C5) − a(C7W-C4direct)| ≤ 0.05: verdict "**the transcription-and-text route may be sufficient**";
  - a(C7W-C4direct) − a(C7W-C5) ≥ 0.10: verdict "**separated speech itself is also harder to transcribe**";
  - otherwise: verdict "**cannot be distinguished**".
- Also reported, for reference only and not part of the verdict: the differences of C7W-C5 from C4-direct under P2 (0.670), C5 (0.395) and C7 plan A (0.255).

## Job (to be confirmed)
- One GPU job, putting Whisper (about 3 GB) and Qwen (loading peak about 21 GiB) on the same A100 40GB.
- Configuration: <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes.
- Expected time: transcription of 524 items + T2 extraction of 524 items, estimated at 12–18 minutes; measurements take precedence.
- Script: `scripts/qwen/qwen_omni_c7w.py` (to be written). Can share a job with the Qwen part of F1.

## Results (2026-11-05; **incomplete**)
- Run record: in the same GPU job 22016221 as F1, **timed out at the 30-minute limit** (17:02:44–17:32:56). Time per part:
  - Whisper: W_C1 39.2 s, W_C4direct 61.6 s, W_C5 60.3 s, all completed;
  - T2: T_C1 221.4 s and T_C4direct 356.8 s completed; **T_C5 did not complete and was not written out**.
- This time Qwen took about 1.9 s per item, versus about 1.1 s before, so the job was slower than estimated. Per the rules, the configuration was not changed and the job was not resubmitted.
- Loading 70.2 s, GPU memory peak during loading 20.83 GiB.
- **The C7-W verdict cannot be computed**, because the T2 output of C7W-C5 is missing.

### Completed conditions
| Condition | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | M | U | Parse failures / refusals |
|---|---|---|---|---|---|---|---|---|---|
| C7W-C1 | 0.565 | 0 | 0.435 | 0.650 | 0.565 | 0.39 | 189 | 9 | 0 / 0 |
| C7W-C4direct | 0.555 | 0.005 | 0.440 | 0.640 | 0.555 | 0.37 | 187 | 11 | 0 / 0 |
| C7W-C5 | — | — | — | — | — | — | — | — | not completed |

### Whisper transcripts (200-sample basis, normalize_v2)
| Group | WER against target sentence | WER against interferer sentence | Items with WER < 0.5 against the target sentence | Closer to target / closer to interferer |
|---|---|---|---|---|
| C1 | 0.075 | 1.045 | 189 | 196 / 0 |
| C4direct | 0.078 | 1.047 | 186 | 196 / 0 |
| C5 | **0.452** | 1.060 | **109** | 153 / 19 |

- Description: on the separated speech of C5, Whisper's WER against the target sentence is 0.45, clearly higher than the 0.078 on the direct reference. This only describes the transcription step and is not the C7-W verdict.

## Rerun of T_C5 (2026-11-05, approved; rules unchanged)
- Reads only the `W_C5/raw.jsonl` already written by job 22016221, i.e. Whisper's transcripts of the C5 target stream; **Whisper is not rerun**. Extraction with the frozen T2 prompt, with input format and decoding settings exactly as in the original job: [transcript.strip() + newline, PROMPT_T2], greedy, max_new_tokens 256.
- Entry point: `scripts/qwen/qwen_omni_t_c5.py` and `.sbatch`, configuration <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes. Additionally records the commit, node name, GPU model and per-item time.
- Output written to `outputs/qwen_omni_c7w_f1_tc5/<jobid>/T_C5/`, without overwriting 22016221.
- Evaluation: create `outputs/qwen_omni_c7w_f1_combined/<jobid>/`, in which F1, R1same, W_*, T_C1 and T_C4direct are symlinks to 22016221 and T_C5 points to the rerun result; then run a complete re-evaluation with `eval_c7w_f1.py`.

## Rerun of T_C5 and complete evaluation (2026-11-05)
- T_C5 rerun: job 22016732, node <NODE>, GPU model **NVIDIA A100-PCIE-40GB** (driver 610.43.02), 20:58:04–21:03:15, exit code 0, MaxRSS 26.1 GB.
  - The commit recorded by the job is `eb85b2e`, with no uncommitted dependency files; the sha256 of the input `W_C5/raw.jsonl` is `c6dee726…f705`. CPU dry-run srun 22016722.
  - Loading 49.0 s; the 200 items took 207.6 s in total, **1.04 s per item on average**, inference GPU memory peak 20.83 GiB. Per-item times are recorded in the `seconds` field of `raw.jsonl`.
  - This ran on the same node as 22016221 but was much faster per item: in 22016221, F1 and R1-same were P2 with audio input, about 1.9 s per item; here it is text-only T2, about 1.04 s per item. The reason for the slowness was not investigated separately.
- Complete evaluation: under `outputs/qwen_omni_c7w_f1_combined/22016732/`, the conditions of 22016221 and the rerun T_C5 are combined with symlinks, and `eval_c7w_f1.py` is run once in full. Results in `outputs/c7w_f1_eval/22016732_full/eval_c7w_f1.json`. The F1 part is exactly the same as the earlier separate computation.

| Condition | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | M | U | Parse failures / refusals |
|---|---|---|---|---|---|---|---|---|---|
| C7W-C1 | 0.565 | 0 | 0.435 | 0.650 | 0.565 | 0.39 | 189 | 9 | 0 / 0 |
| C7W-C4direct | 0.555 | 0.005 | 0.440 | **0.640** | 0.555 | 0.37 | 187 | 11 | 0 / 0 |
| **C7W-C5** | 0.290 | 0.005 | 0.705 | **0.340** | 0.290 | 0.13 | 109 | 87 | 3 / 3 |
| C5 (P2, reference) | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 | 82 | 0 / 0 |

Basis of M: for C7W-C5 counted against either talker; for C7W-C1 and C7W-C4direct only against the target.

- **C7-W verdict (frozen rules)**: a(C7W-C4direct) − a(C7W-C5) = 0.640 − 0.340 = **0.30**, ≥ 0.10 → **separated speech itself is also harder to transcribe**.
- For reference only (not part of the verdict): C7W-C5 is 0.055 below C5 under P2, 0.33 below C4-direct under P2, and 0.085 above C7 plan A.
