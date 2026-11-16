English translation of fasnet_single_talker_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：fasnet_single_talker_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# F1 v1: FaSNet-TAC on single-talker full reverberation, and R1-same (fasnet-single-talker-v1; 2026-11-05; rules written before the run and not changed after the results were seen)

Purpose: separate "distortion caused by FaSNet processing itself" from "residual two-talker overlap". No model fine-tuning, no new front end, no change to mix_v1, the P2 prompt, binding-score-v1, existing results or verdicts.

## Input
- For each sample of mix_v1 (200), take only the full-reverberation 6-mic image of the **target speaker** in that sample.
  - Azimuths: the target A of s1 is at position 0 (θ1), the target B of s2 is at position 0 (θ1).
  - Recomputed in memory with the functions of `mix_slurp_v1`, using the gains and rir_start in the mix_v1 manifest and the offsets in the list.
  - **No interferer, no noise**.
- Note: R1 uses the RIR and position of the first appearance of the target recording in mix_v1 (see c3-factor-diag-v1), whereas F1 uses each sample's own RIR and position, so their RIRs are not exactly the same. The evaluation reports the number of samples whose RIR and position are exactly the same for both, and gives the results for that subset separately.

## Processing
1. FaSNet-TAC inference: checkpoint read-only, inference code the same as `frontend_mix_v1.py` (std normalisation → model → × scale), run with srun on a CPU node, tfgridnet environment.
2. Stream selection: **select the stream with the higher energy (RMS)**, without using reference signals.
3. Polarity: zero-lag correlation with the mic 0 input; invert if the correlation is negative, the same as the v3 rule.
4. The selected stream (8 kHz, float32) is written to `outputs/fasnet_single_talker_v1/streams/`; written only to outputs, not to data.
5. Into Qwen: `resample_poly(x, 2, 1)[:2T]` → −30 dBFS → frozen P2, without the direction instruction; decoding settings as for P2, and the flow the same as C5.

## Metrics
- Using `eval_c4_v1.score` and `summ`, unchanged. D_T and D_I are computed from each sample's own pair.
- Reported: target only, interferer only, neither, target-any, all target values hit, pair-flip accuracy, M, U. M uses the same basis as R1, considering only the target's own WER < 0.5.
- Also reported: the SI-SDR between the selected stream and the target direct reference (descriptive only), and the energy ratio of the two streams.

## R1-same (added 2026-11-05, as the direct control for F1)
- Each mix_v1 sample uses the target's full-reverberation **mic 0** signal, with RIR, position, gain and offset all the same as F1, no interferer and no noise, **not passed through FaSNet**.
- Processing: 8 kHz mic 0 full image → `resample_poly(x, 2, 1)[:2T]` → −30 dBFS → frozen P2. Same flow as F1 and C5.
- Metrics as for F1; M considers only the target.

## Verdict (empirical, not significance tests; a = target-any over 200 samples; C5 = 0.395)
The main verdict uses **all 200 samples** and compares with **R1-same**:
- |a(F1) − a(R1-same)| ≤ 0.05, and a(R1-same) − a(C5) ≥ 0.10: verdict "**problems related to residual overlap are more likely**";
- a(R1-same) − a(F1) ≥ 0.10, and |a(F1) − a(C5)| ≤ 0.05: verdict "**FaSNet processing distortion is more likely**";
- otherwise: verdict "**in between, or cannot be distinguished**".
- **Check only, not part of the verdict**: the original R1 (0.575, RIR from the group of the recording's first appearance); and whether R1-same agrees with the original R1 on the 63 samples whose RIR and position are exactly the same as the original R1.
- Diagnosis only, not written as causal.

## Jobs (to be confirmed)
- First step, FaSNet inference: srun, <SLURM_ACCOUNT>, <CPU_PARTITION>, 4 CPU, 16 GB, 40 minutes.
- Second step, Qwen: F1, R1-same and C7-W in the same GPU job, configured as <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes.
- Scripts: `scripts/sms_wsj/fasnet_single_talker.py`, `scripts/qwen/eval_f1.py` (to be written).

## Results (2026-11-05)
- Run records:
  - FaSNet inference for F1: srun 22016218 (<CPU_PARTITION>, tfgridnet environment). Dry-run srun 22016217, script commit `5fd39c1`.
  - Qwen part: GPU job 22016221, <NODE>, started 17:02:44, **reached the 30-minute limit at 17:32:56 and was judged timed out** (TIMEOUT, MaxRSS 30.0 GB). The commit recorded by the job is `5fd39c1`, with no uncommitted dependency files.
  - F1 (381.9 s) and R1-same (386.4 s) had completed and been written out before the timeout. This section uses only these two conditions.
  - Evaluation: with `eval_c4_v1.score` and `summ`, the verdict logic the same as `eval_c7w_f1.py`. Because T_C5 of C7-W in the job did not finish, the original evaluation script could not run as a whole, so the F1 part was computed separately on the login node with the same functions. Results in `outputs/c7w_f1_eval/22016221_partial/eval_partial.json`.
- Stream selection in F1: of the 200 samples, stream 0 was selected 15 times, and all 15 were inverted by the polarity rule; the median RMS ratio of the two streams is 4.9e-8, i.e. one stream is essentially zero; the median SI-SDR of the selected stream relative to the target direct reference is 8.86 dB.

| Condition | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | M (target WER < 0.5) | U | Parse failures / refusals |
|---|---|---|---|---|---|---|---|---|---|
| **F1** (single-talker full reverberation → FaSNet) | 0.525 | 0 | 0.475 | **0.610** | 0.525 | 0.31 | 179 | 21 | 0 / 0 |
| **R1-same** (mic 0 with the same RIR, not through FaSNet) | 0.490 | 0 | 0.510 | **0.575** | 0.490 | 0.26 | 174 | 23 | 0 / 0 |
| C5 (reference) | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 (either talker) | 82 | 0 / 0 |
| Original R1 (check only) | 0.490 | 0 | 0.510 | 0.575 | 0.490 | 0.28 | 176 | 21 | 0 / 0 |

- **Verdict (frozen rules, 200 samples)**: |a(F1) − a(R1-same)| = 0.035 ≤ 0.05, and a(R1-same) − a(C5) = 0.18 ≥ 0.10 → **problems related to residual overlap are more likely**. This is only a diagnosis, not a causal conclusion.
- Check: on the 63 samples whose RIR and position are the same as the original R1, R1-same is 0.571 and the original R1 0.587; 49/63 have exactly identical raw outputs. The reason they are not all identical is the level: R1 sets the gain from the active-frame level of the 8 kHz early image, R1-same from the active-frame level of the 16 kHz full-reverberation signal; the two normalise levels differently.
