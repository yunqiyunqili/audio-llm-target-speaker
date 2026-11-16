English translation of separation_quality_diag_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：separation_quality_diag_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Separation quality diagnosis v1 (separation-quality-diag-v1; 2026-11-05; rules written before computation and not changed after the results were seen; read-only)

Purpose: look at the relationship between the downstream failures of C5 FaSNet-TAC and separation quality. **Diagnosis only; correlation is not written as causation.** Qwen is not run, the front end is not rerun, and existing results are not changed.

## Object
- The 200 samples of C5 FaSNet-TAC (job 22015337). Target stream = the stream selected by C5, used after the C5 polarity correction, consistent with the signal Qwen actually received (before upsampling and level adjustment).
- References: the direct references of the target and the interferer (`outputs/frontend_mix_v1_eval_v2/direct_refs/`, 8 kHz, already checked against the official definition).

## Separation quality metrics (per sample, 8 kHz)
- First make y, d_T and d_I zero-mean, then project y onto span{d_T, d_I} by least squares: y = a·d_T + b·d_I + e_r. Let e_t = a·d_T, e_i = b·d_I.
  - **SI-SDR_T**: the value of y relative to d_T computed with `tfgridnet/objectives.py: sisdr`, as in v2.
  - **Interference residual ratio IRR** = 10·log10(‖e_t‖² / ‖e_i‖²), in dB. Larger means less residual interference.
  - **Residual artefact ratio RAR** = 10·log10(‖e_t‖² / ‖e_r‖²), in dB. Larger means less residual unrelated to both references (distortion, residual reverberation, noise).
- Stratum boundaries (fixed in advance):
  - SI-SDR_T: < 5, [5,10), ≥ 10 dB;
  - IRR: < 10, [10,20), ≥ 20 dB;
  - RAR: < 5, [5,10), ≥ 10 dB.

## Downstream metrics (same as C4/C5, calling `eval_c4_v1.score` as is)
- Per stratum, report: target only, interferer only, neither, target-any, M (WER < 0.5 against either talker), U (unrelated to both sentences).
- Each stratum also gives the C4-direct values for the same samples.
- Item-by-item comparison of C5 and C4-direct: a 2×2 table on target-any (both hit, only C4-direct hits, only C5 hits, neither hits), reported once overall and once for each SI-SDR_T stratum.

## Interpretation
- Only the differences between strata and the 2×2 tables are described; **no causal judgement and no threshold verdict**.
- The existing conclusion stays qualified: "separation-related degradation is an important bottleneck". This diagnosis does not judge the separate contributions of residual interference and distortion; IRR and RAR are descriptive metrics only.

## Implementation
- First step (tfgridnet environment, CPU, srun): `scripts/sms_wsj/sep_quality_metrics.py` → `outputs/separation_quality_diag_v1/metrics.jsonl`.
- Second step (read-only, pure Python): `scripts/qwen/sep_quality_diag.py` → `outputs/separation_quality_diag_v1/diag.json`.

## Results (2026-11-05; descriptive only, **correlation is not causation**)
- Run records:
  - First step: srun 22015392 (<CPU_PARTITION>, tfgridnet environment). When the first srun failed (the time of the error was not recorded; it was still failing at 15:34:12 at the latest and recovered at 15:34:55), Slurm commands on the login node reported a configuration error ("Unable to process configuration file"); it was run only after recovery.
  - Second step: run on the login node, read-only.
  - Script commit `080e1a5`. Results in `outputs/separation_quality_diag_v1/{metrics.jsonl,diag.json}`.
- Metric distributions (200 samples, min / median / max):
  - SI-SDR_T: −16.1 / 5.95 / 14.1 dB;
  - IRR: −6.2 / 30.0 / 87.0 dB;
  - RAR: −15.5 / 6.1 / 14.2 dB.

### Overall (200 samples)
| | Target only | Interferer only | Neither | target-any | M | U |
|---|---|---|---|---|---|---|
| C5 FaSNet-TAC | 0.340 | 0.010 | 0.650 | 0.395 | 111 | 82 |
| C4-direct | 0.565 | 0 | 0.435 | 0.670 | 185 | 15 |

Item-by-item 2×2 table (target-any): both hit 75, only C4-direct hits 59, only C5 hits 4, neither hits 62.

### By stratum (C5 target only / interferer only / neither / target-any / M / U; and C4-direct target-any; 2×2 order: both hit / only C4-direct / only C5 / neither)
| Metric and stratum | Samples | C5 | C4-direct target-any | 2×2 |
|---|---|---|---|---|
| SI-SDR_T < 5 dB | 71 | 0.268 / 0.028 / 0.704 / 0.338 / 27 / 38 | 0.690 | 22 / 27 / 2 / 20 |
| SI-SDR_T [5,10) dB | 114 | 0.377 / 0 / 0.623 / 0.430 / 73 / 40 | 0.658 | 47 / 28 / 2 / 37 |
| SI-SDR_T ≥ 10 dB | 15 | 0.400 / 0 / 0.600 / 0.400 / 11 / 4 | 0.667 | 6 / 4 / 0 / 5 |
| IRR < 10 dB | 16 | 0 / 0.125 / 0.875 / 0.188 / 2 / 11 | 0.625 | 2 / 8 / 1 / 5 |
| IRR [10,20) dB | 32 | 0.250 / 0 / 0.750 / 0.312 / 14 / 15 | 0.719 | 10 / 13 / 0 / 9 |
| IRR ≥ 20 dB | 152 | 0.395 / 0 / 0.605 / 0.434 / 95 / 56 | 0.664 | 63 / 38 / 3 / 48 |
| RAR < 5 dB | 69 | 0.275 / 0.029 / 0.696 / 0.333 / 26 / 37 | 0.696 | 21 / 27 / 2 / 19 |
| RAR [5,10) dB | 116 | 0.371 / 0 / 0.629 / 0.431 / 74 / 41 | 0.655 | 48 / 28 / 2 / 38 |
| RAR ≥ 10 dB | 15 | 0.400 / 0 / 0.600 / 0.400 / 11 / 4 | 0.667 | 6 / 4 / 0 / 5 |

### Description (no causal judgement, no thresholds)
- Residual interference is very low in most samples: 152/200 have IRR ≥ 20 dB, median 30 dB. The median of RAR, however, is only 6.1 dB, close to SI-SDR_T. That is, the residual energy in the target stream that is unrelated to both direct references is far larger than the residual interference energy.
- The downstream performance of C5 improves slightly as SI-SDR_T and RAR rise (target-any from 0.33 to 0.43), but even in the SI-SDR_T ≥ 10 dB stratum (15 samples) it is only 0.40, whereas C4-direct on the same samples is 0.67.
- In the IRR < 10 dB stratum (16 samples), C5 is only 0.19, and "interferer only" appears (0.125).
- Item-by-item comparison: 59 samples are hit only by C4-direct, and only 4 only by C5.
- The existing conclusion is unchanged: separation-related degradation is an important bottleneck. This diagnosis **does not judge** the separate contributions of residual interference and distortion; "larger residual energy" is only a description of the metrics.
