English translation of c6_delta_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c6_delta_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C6δ v1: end-to-end results with simulated error in the query azimuth, and simple abstention (c6-delta-v1; 2026-11-05; rules written before the run and not changed after the results were seen)

> **The azimuth error here is an artificially added fixed offset δ (simulated error), not the error of real users or real localisers.** δ = 0 is C5, which depends on the oracle azimuth. No downloads, no model fine-tuning, no new front end; mix_v1, the P2/T2 prompts, binding-score-v1, existing results and the v1/v2/v3 verdicts are not changed.

## C6δ
- Front end: the existing FaSNet-TAC outputs. Stream selection: the frozen v3 rule (polarity correction → `doa` → take the nearest stream, stream 0 on a tie). Query azimuth θ_query = wrap(θ1 + δ), δ ∈ {0, +10, −10, +20, −20}. The estimated directions of the two streams are taken directly from `outputs/c5_v1/selection.jsonl`, the same as direction-error-selection-v1.
- Qwen outputs are needed for both streams:
  - the stream selected by C5: the existing outputs of job 22015337 are used directly, not rerun;
  - **the other stream** (1 − the stream selected by C5): run additionally with P2 in this round. Processing exactly as for C5: correct by the v3 polarity flag → `resample_poly(x, 2, 1)[:2T]` → −30 dBFS → frozen P2, without the direction instruction → decoding settings as for P2.
- For each δ: each sample takes the Qwen output of its selected stream, computed with `eval_c4_v1.score` and `summ`. Report target only, interferer only, neither, target-any, all target values hit, pair-flip accuracy, and M and U.
- Strata: reported by the four angle strata. Additionally **weighted by the natural angle distribution of cv_dev93**: [0,15) 78, [15,30) 96, [30,60) 155, [60,180] 653, 982 in total; weighted value = Σ(stratum weight × stratum metric). Pair-flip accuracy is not weighted; only the unweighted value is reported.
- Consistency: the results at δ = 0 must be exactly identical to C5 (eval_c5_v1).

## Simple abstention
- margin = |wrap(θ_query − θ̂_other)| − |wrap(θ_query − θ̂_selected)|, in degrees, ≥ 0 by definition. When margin < τ, output "cannot determine" and do not use the answer for that sample.
- This rule **uses only the estimated directions and the query azimuth, neither reference signals nor speaker identity**.
- Split into two halves by group: sort the 100 group numbers and shuffle with `random.Random(2026).shuffle`; the first 50 groups are used to choose the threshold (dev), the last 50 for evaluation (eval). The two swapped samples of each group are in the same half.
- Choosing τ (dev half only, pooling the samples of the five δ): among τ ∈ {0, 1, …, 90}°, take the **smallest** τ for which the stream-selection error rate among answered samples is ≤ 0.05. If none exists, take the τ that minimises the error rate (the smaller τ on ties). Whether stream selection is wrong is judged by direct-reference matching; this is used only for choosing the threshold and evaluating; the abstention rule itself does not use references.
- On the eval half, for each δ, report: coverage (proportion answered), the stream-selection error rate among answered samples, the proportion of interferer-only among answered samples, and the proportion of target-any among answered samples; with the τ = 0 (no abstention) control alongside.
- Risk–coverage curves: on the eval half, for τ ∈ {0, 1, …, 90}°, coverage on the x-axis and the stream-selection error rate of answered samples on the y-axis, one line per δ; the figure is saved as `docs/figures/c6_risk_coverage_v1.png`.
- No verdict thresholds; descriptive only.

## Implementation
- Qwen inference on the other stream runs in the same GPU job as C7: `scripts/qwen/qwen_omni_c6c7.py` and `.sbatch`.
- Evaluation: `scripts/qwen/eval_c6_delta.py`, results in `outputs/c6_delta_v1/`.

## Results (2026-11-05)
> **δ is an artificially added fixed offset (simulated error), not the error of real users or localisers. δ = 0 is C5 (depends on the oracle azimuth).**
- Run records:
  - GPU job 22015445: <NODE>, 16:00:13–16:14:38, exit code 0, MaxRSS 27.0 GB. The commit recorded by the job is `4caeb4b`, with no uncommitted dependency files. CPU dry-run srun 22015398.
  - Inference on the other stream (OTHER) took 223.8 s.
  - Evaluation: `scripts/qwen/eval_c6_delta.py`, run on the login node, reading raw outputs only. Results in `outputs/c6_delta_v1/eval_c6_delta.json`.
- The results at δ = 0 are exactly identical to C5; the script asserts this.

### End-to-end results for each δ (200 samples)
| δ | Stream-selection error rate | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | target-any / interferer-only weighted by the natural cv_dev93 distribution |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.010 | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 0.408 / 0.003 |
| +10 | 0.220 | 0.290 | 0.075 | 0.635 | 0.335 | 0.290 | 0.10 | 0.387 / 0.026 |
| −10 | 0.130 | 0.300 | 0.055 | 0.645 | 0.335 | 0.300 | 0.10 | 0.388 / 0.018 |
| +20 | 0.405 | 0.245 | 0.140 | 0.615 | 0.280 | 0.245 | 0.09 | 0.359 / 0.059 |
| −20 | 0.215 | 0.265 | 0.080 | 0.655 | 0.300 | 0.265 | 0.09 | 0.368 / 0.033 |

"Both" is 0 for every δ. The weights are the natural angle distribution of cv_dev93: [0,15) 78, [15,30) 96, [30,60) 155, [60,180] 653.

### By angle stratum (target-any / interferer-only)
| δ | [0,15) | [15,30) | [30,60) | [60,180] |
|---|---|---|---|---|
| 0 | 0.34 / 0.04 | 0.30 / 0 | 0.54 / 0 | 0.40 / 0 |
| +10 | 0.20 / 0.16 | 0.20 / 0.14 | 0.54 / 0 | 0.40 / 0 |
| −10 | 0.14 / 0.18 | 0.26 / 0.04 | 0.54 / 0 | 0.40 / 0 |
| +20 | 0.20 / 0.16 | 0.08 / 0.28 | 0.44 / 0.12 | 0.40 / 0 |
| −20 | 0.14 / 0.18 | 0.22 / 0.06 | 0.44 / 0.08 | 0.40 / 0 |

### Simple abstention (using neither reference signals nor speaker identity)
- Split into two halves by group: dev 50 groups, eval 50 groups; the group lists are in the json.
- On dev the rule selects **τ = 17°**: the smallest τ making the stream-selection error rate of answered samples ≤ 0.05. On dev (five δ pooled), coverage 0.524, stream-selection error rate 0.042.
- On eval (100 samples per δ):

| δ | Coverage (τ = 17°) | Answered samples: stream-selection error rate / interferer-only / target-any | No abstention (τ = 0): stream-selection error rate / interferer-only / target-any |
|---|---|---|---|
| 0 | 0.71 | 0 / 0 / 0.451 | 0 / 0.01 / 0.42 |
| +10 | 0.53 | 0 / 0 / 0.453 | 0.19 / 0.07 / 0.36 |
| −10 | 0.70 | 0 / 0 / 0.457 | 0.12 / 0.04 / 0.37 |
| +20 | 0.50 | 0.14 / 0.04 / 0.36 | 0.40 / 0.15 / 0.27 |
| −20 | 0.63 | 0 / 0 / 0.46 | 0.16 / 0.05 / 0.36 |

- Risk–coverage curves: `docs/figures/c6_risk_coverage_v1.png` (eval half, τ from 0 to 90°, "+" marks τ = 17°).
- Description:
  - With abstention, stream-selection errors and interferer-only among answered samples drop clearly, at the cost of coverage falling to 0.50–0.71.
  - target-any of answered samples is between 0.36 and 0.46, still limited by separation quality (see C5 and the separation quality diagnosis).
  - At δ = +20, the stream-selection error rate is still 0.14 even with abstention.
