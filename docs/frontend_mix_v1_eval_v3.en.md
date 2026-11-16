English translation of frontend_mix_v1_eval_v3.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：frontend_mix_v1_eval_v3.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# frontend_mix_v1 evaluation v3: polarity correction before direction estimation (frontend-mix-v1-eval-v3; 2026-11-05; rules written before computation and not changed after the results were seen)

> **Note**: this correction was proposed only after the mix_v1 results had been seen (stream 0 of FaSNet-TAC has inverted polarity, see `docs/c4_v1.md`), so its results on mix_v1 **may be optimistic**. The check on cv_dev93 took no part in designing the correction and serves as an external check.

## Scope
- Only the existing `data/mix_v1/separated_v1/` outputs are used; **the mix_v1 front ends are not rerun**. C5 is not run, no new front end is added, and no direction matching v4 is written. mix_v1, existing results and the v1/v2 verdicts are not changed.
- FaSNet-TAC is the main object of evaluation; BLSTM+MVDR uses the same rule, as a control only.

## Polarity-correction rule (no reference signals, no hard-coded output-stream index)
- For each output stream y_k of each sample, compute its zero-lag correlation with the mic 0 observation, c_k = Σ_t y_k(t) · x_0(t). If c_k < 0, invert y_k; otherwise leave it unchanged.
- The rest of direction estimation is exactly the same as v1/v2 (`eval_frontend_mix_v1.doa`): take the stream minimising |wrap(θ̂_k − θ1)|, **stream 0 when the directions are equal**.
- Accuracy uses **direct-reference matching** (`ref_stream_direct` of v2) as the standard. SI-SDR is unaffected by sign, so reference matching need not be recomputed.

## Verdict (threshold unchanged)
- Direction-matching accuracy is computed only on the 150 samples whose angle is not in [0,15): if ≥ 0.85, the front end may proceed to C5 under v3; otherwise it does not proceed to C5. [0,15) is only reported. The threshold is empirical.
- This round does not proceed to C5. Even if v3 judges that it may proceed, only the verdict is recorded.

## Reported on mix_v1
- Direction-matching accuracy (all 200, and the 150 excluding [0,15)) and the failed samples;
- strata (angle, T60, active overlap ratio, full-image SIR);
- direction errors of both streams: target stream against θ1, interferer stream against θ2; also reported by output-stream index;
- proportion of polarity corrections: the inversion proportion per output-stream index, and whether the correlation with the direct reference is positive after correction.

## External check: 50 official mixtures of SMS-WSJ cv_dev93
- The 50 groups are drawn with `random.Random(2026).sample(sorted(cv_dev93 ids), 50)`, the same as B and C of frontend_sanity_v1.
- This time both front ends need to be **rerun** on cv_dev93: with srun on a CPU node, inference code the same as frontend_mix_v1; outputs only in memory, no audio written. The mix_v1 front-end outputs are not rerun.
- Each group yields 2 "target samples": when the target is source 0, θ_t = az_rel of source 0; likewise when the target is source 1. Reference matching uses PIT with the official direct references.
- Reported:
  - the zero-lag correlation of each stream with its own direct reference, per output-stream index;
  - whether the same polarity flip exists (proportion with correlation < 0);
  - direction-matching accuracy before and after correction (all, and excluding [0,15)).
- The cv_dev93 results do not change the mix_v1 verdict.

## Implementation
- `scripts/sms_wsj/eval_frontend_mix_v1_v3.py` (mix_v1, CPU, tfgridnet environment);
- `scripts/sms_wsj/polarity_check_cv_dev93.py` (cv_dev93, CPU, tfgridnet environment, front-end inference in memory);
- results written to `outputs/frontend_mix_v1_eval_v3/`.

## Results (2026-11-05)
> Again: this correction was proposed only after the mix_v1 results had been seen, so the numbers on mix_v1 **may be optimistic**. cv_dev93 is the external check.
- Run record: srun 22015151 (<CPU_PARTITION>, tfgridnet environment), script commit `e3d7d04`. Results in `outputs/frontend_mix_v1_eval_v3/{mix_v1,cv_dev93}/`. Front-end inference on cv_dev93 was done only in memory; no audio was written.

### mix_v1 (200 samples, existing separated_v1 outputs)
| | FaSNet-TAC v3 | FaSNet-TAC v2 (reference) | BLSTM+MVDR v3 (control) |
|---|---|---|---|
| Direction-matching accuracy (all 200) | 0.99 | 0.50 | 0.735 |
| **Direction-matching accuracy (excluding [0,15), 150)** | **1.00** | 0.513 | 0.713 |
| Verdict under v3 | **may proceed to C5** (only recorded this round; C5 not entered) | — | does not proceed to C5 |
| Failed samples (excluding [0,15)) | 0 | 73 | 43 |
| Polarity-correction proportion: stream 0 / stream 1 | **1.00 / 0.00** | — | 0.00 / 0.00 |
| Proportion with positive correlation to the direct reference after correction: stream 0 / stream 1 | 1.00 / 1.00 | — | 1.00 / 1.00 |
| Median direction error target stream / interferer stream | 0.81° / 0.76° | 77.9° / 2.0° | 2.9° / 2.3° |
| Median direction error stream 0 / stream 1 | 0.81° / 0.76° | — | 2.7° / 2.3° |
| Both streams same direction / differing by ≤ 5° | 0 / 0.115 | — | 0.335 / 0.805 |

- FaSNet-TAC fails on only 2 of all 200 samples, both in the [0,15) stratum: g006_s1 (θ1 = −39.7°, θ2 = −42.0°) and g074_s1 (θ1 = −146.6°, θ2 = −147.6°).
- Direction-matching accuracy by stratum (FaSNet-TAC v3 / BLSTM+MVDR v3):
  - angle [0,15), [15,30), [30,60), [60,180]: 0.96 / 1.00 / 1.00 / 1.00, control 0.80 / 0.78 / 0.70 / 0.66;
  - T60 [0.2,0.3), [0.3,0.4), [0.4,0.5]: 1.00 / 1.00 / 0.98, control 0.75 / 0.80 / 0.69;
  - active overlap ratio [0,0.5), [0.5,0.75), [0.75,0.9), [0.9,1]: 1.00 / 0.99 / 1.00 / 0.96, control 0.78 / 0.79 / 0.68 / 0.75;
  - full-image SIR < −0.5, [−0.5,0.5], > 0.5 dB: 1.00 / 0.99 / 1.00, control 0.84 / 0.74 / 0.63.

### cv_dev93 external check (50 official mixtures, 2 target samples each, 100 in total; 92 excluding [0,15))
| | FaSNet-TAC | BLSTM+MVDR |
|---|---|---|
| Proportion with negative correlation to its own direct reference: stream 0 / stream 1 | **1.00 / 0.00** | 0.00 / 0.00 |
| Median correlation: stream 0 / stream 1 | **−0.94** / +0.95 | +0.66 / +0.61 |
| Polarity-correction proportion: stream 0 / stream 1 | 1.00 / 0.00 | 0.00 / 0.00 |
| Direction-matching accuracy, before → after correction (all 100) | 0.64 → **1.00** | 0.96 → 0.96 |
| Direction-matching accuracy, before → after correction (excluding [0,15)) | 0.652 → **1.00** | 0.957 → 0.957 |
| Median target-stream direction error, before → after correction | 33.7° → 0.56° | 1.02° → 1.02° |

- The 50 cv_dev93 groups by angle stratum: [0,15) 4 groups, [15,30) 7, [30,60) 9, [60,180] 30, i.e. the natural distribution.
- Conclusion (description): the inverted polarity of FaSNet-TAC stream 0 also exists on the official SMS-WSJ data and **is not specific to mix_v1**. After correcting polarity with the zero-lag correlation, direction matching is close to perfect on both datasets.
- BLSTM+MVDR has no polarity problem. Its direction-matching accuracy is 0.957 on cv_dev93 and 0.713 on mix_v1.
- The v1 and v2 results and verdicts remain unchanged; the cv_dev93 results do not change the mix_v1 verdict.
