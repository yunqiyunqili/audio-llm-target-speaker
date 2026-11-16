English translation of c4_gap_diag_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c4_gap_diag_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C4 gap diagnosis v1 (c4-gap-diag-v1; 2026-11-05; rules written before the run and not changed after the results were seen)

Background: target-any of C4 FaSNet-TAC is 0.395 and of R1 0.575, a difference of 0.18, judged per `docs/c4_v1.md` as "distortion is still the main obstacle". This is a diagnosis only: C5 is not run, and P2, binding-score-v1, existing results and verdicts are not changed.

## A. Read-only analysis: the 129 "neither" items of C4 FaSNet-TAC
- Stratified by the SI-SDR of the FaSNet-TAC target stream (direct reference; from `outputs/frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl`): < 5, [5,10), [10,15), ≥ 15 dB.
- Per stratum, report: count, proportion of all samples in that SI-SDR stratum, M (WER < 0.5 against either talker), U (unrelated to both sentences), and whether the transcript is closer to the target or the interferer (the definition of "clear choice" is the same as in `c3_output_diag.py`).

## B. C4-direct (GPU)
- Input: the **direct reference** of the target speaker in the same 200 samples (`outputs/frontend_mix_v1_eval_v2/direct_refs/`, 8 kHz, already checked against the official definition).
- Processing exactly as for C4: `resample_poly(x, 2, 1)[:2T]` → active-frame RMS normalised to −30 dBFS (512-point frames, peak −40 dB) → frozen P2, without the direction instruction → decoding settings as for P2.
- Job configuration: <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes. The working tree must be clean before job submission.

## Comparison and interpretation rules (a = target-any over 200 samples, m = M / 200)
- Three conditions are compared: C4 FaSNet-TAC (job 22014545), C4-direct, R1 (job 22014155, judged per sample by pair). The metrics are the four binding-score-v1 class proportions, target-any, all target values hit, and M, U.
- Define d_sep = a(C4-direct) − a(C4 FaSNet), d_chain = a(R1) − a(C4-direct).
- Interpretation rules:
  - d_sep ≥ 0.10: verdict "**separation residue and distortion are the main source of the C4 gap**";
  - |d_chain| ≥ 0.10: verdict "**the difference between the direct-sound chain and the R1 chain (direct sound without reverberation, level definition) itself has an effect**", with the sign reported;
  - both < 0.10: verdict "**cannot be distinguished**".
  - Both can hold at the same time.
- The thresholds are empirical, not significance tests.

## Implementation
- `scripts/qwen/c4_neither_by_sisdr.py` (A; read-only);
- `scripts/qwen/qwen_omni_c4_direct.py` and `.sbatch` (B);
- `scripts/qwen/eval_c4_gap.py` (comparison).
- Results are written to `outputs/qwen_omni_c4_direct/<jobid>/`, `outputs/c4_gap_diag/`.

## Results (2026-11-05)
- Run records:
  - C4-direct: GPU job 22015148, <NODE>, 14:44:32–14:49:48, exit code 0, MaxRSS 26.6 GB. The commit recorded by the job is `e3d7d04`, with no uncommitted dependency files; the combined sha256 of direct_refs is recorded in `config.json`. CPU dry-run: srun 22015015.
  - Evaluation: `scripts/qwen/eval_c4_gap.py`, run on the login node, reading raw outputs only.
- One difference between implementation and plan: part A was not written as a separate `c4_neither_by_sisdr.py` but merged into `eval_c4_gap.py`; the rules are unchanged.
- Results in `outputs/c4_gap_diag/22015148/eval_c4_gap.json`.

### A. The 129 "neither" items of C4 FaSNet-TAC, stratified by target-stream SI-SDR (direct reference)
| SI-SDR stratum | Samples in stratum | Of which neither | Proportion of stratum | M | U | Transcript clearly chose target / no clear choice |
|---|---|---|---|---|---|---|
| < 5 dB | 71 | 49 | 0.69 | 9 | 37 | 9 / 40 |
| [5,10) dB | 114 | 71 | 0.62 | 31 | 40 | 31 / 40 |
| [10,15) dB | 15 | 9 | 0.60 | 5 | 4 | 4 / 5 |
| ≥ 15 dB | 0 | — | — | — | — | — |

- Among these 129, no transcript clearly chose the interferer.

### B. C4 FaSNet-TAC, C4-direct and R1 compared (200 samples)
| Condition | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | M | U | Parse failures / refusals |
|---|---|---|---|---|---|---|---|---|---|
| C4 FaSNet-TAC | 0.340 | 0.015 | 0.645 | 0.395 | 0.340 | 0.11 | 112 | 81 | 0 / 0 |
| C4-direct | 0.565 | 0 | 0.435 | **0.670** | 0.565 | 0.36 | 185 | 15 | 0 / 0 |
| R1 | 0.490 | 0 | 0.510 | 0.575 | 0.490 | 0.28 | 176 | 21 | 0 / 0 |

"Both" is 0 for all three conditions.

### Verdict (frozen rules)
- d_sep = a(C4-direct) − a(C4 FaSNet) = **+0.275**, ≥ 0.10 → **separation residue and distortion are the main source of the C4 gap**.
- d_chain = a(R1) − a(C4-direct) = **−0.095**, absolute value < 0.10 → the difference between the direct-sound chain and the R1 chain does not reach the threshold; C4-direct is 0.095 above R1, recorded only.
