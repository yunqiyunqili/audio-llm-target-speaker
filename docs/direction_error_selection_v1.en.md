English translation of direction_error_selection_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：direction_error_selection_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Stream-selection accuracy under query-azimuth error v1 (direction-error-selection-v1; 2026-11-05; rules written before computation and not changed after the results were seen)

## Purpose
C5 uses the true azimuth θ1. Here we check whether the frozen v3 stream-selection rule can still select correctly when the query azimuth has an error. Diagnosis only: Qwen is not run, no new front end is added, and the mix_v1 results and v1/v2/v3 results or verdicts are not changed.

## Method
- Input: the existing FaSNet-TAC and BLSTM+MVDR outputs (`data/mix_v1/separated_v1/`), 200 samples.
- Stream-selection rule: **the frozen v3 rule**, i.e. zero-lag correlation polarity correction → `doa` → take the stream with the smallest |wrap(θ̂_k − θ_query)|, stream 0 on a tie.
  - The estimated directions θ̂_k of the two streams do not depend on the query azimuth, so the values already computed by v3 are used directly: for FaSNet-TAC from `outputs/c5_v1/selection.jsonl`, for BLSTM+MVDR `doa_v3` from `outputs/frontend_mix_v1_eval_v3/mix_v1/eval_v3_per_sample.jsonl`.
- Query azimuth θ_query = wrap(θ1 + δ), δ ∈ {0, +10, −10, +20, −20} degrees.
- Accuracy: the proportion where the selected stream agrees with the direct-reference matching (`ref_stream_direct` of v2). The direct reference is used only for evaluation.
- Consistency requirement: the FaSNet-TAC selection at δ = 0 must agree **item by item** with the C5 selection (`outputs/c5_v1/selection.jsonl`); otherwise stop.
- Strata: angle ([0,15), [15,30), [30,60), [60,180]), T60 ([0.2,0.3), [0.3,0.4), [0.4,0.5]), and all samples excluding [0,15).
- 95% confidence intervals: bootstrap with the group as the unit (the two swapped samples resampled together), 10,000 times, `numpy default_rng(2026)`.

## Verdict (based mainly on FaSNet-TAC; empirical)
Let acc_±20 = min(accuracy at δ = +20, accuracy at δ = −20), computed on the 150 samples excluding [0,15).
- acc_±20 ≥ 0.95: verdict "**direction error is not the current bottleneck of stream selection**";
- acc_±20 < 0.85: verdict "**direction uncertainty is an important bottleneck**";
- otherwise: verdict "**matters only under some conditions**".
- The [0,15) and [15,30) strata are reported separately and take no part in the overall verdict.
- BLSTM+MVDR is computed with the same method, as a control only, with no verdict.

## Implementation
- `scripts/sms_wsj/direction_error_selection.py` (reads only existing json files, pure Python computation); results in `outputs/direction_error_selection_v1/`.

## Results (2026-11-05; read-only computation; results in `outputs/direction_error_selection_v1/direction_error_selection.json`)
- Computation: pure Python reading the existing json, run on the login node. Script commit `080e1a5`.
- At δ = 0, the FaSNet-TAC selection agrees **item by item** with C5; the script asserts this for every item.

### FaSNet-TAC (accuracy [95% confidence interval])
| δ | All 200 | Excluding [0,15) (150) | [0,15) | [15,30) | [30,60) | [60,180] |
|---|---|---|---|---|---|---|
| 0 | 0.990 [0.97,1.00] | 1.000 [1.00,1.00] | 0.96 [0.90,1.00] | 1.00 | 1.00 | 1.00 |
| +10 | 0.780 [0.70,0.85] | 0.900 [0.83,0.96] | 0.42 [0.24,0.60] | 0.70 [0.52,0.86] | 1.00 | 1.00 |
| −10 | 0.870 [0.81,0.93] | 0.967 [0.92,1.00] | 0.58 [0.40,0.76] | 0.90 [0.78,1.00] | 1.00 | 1.00 |
| +20 | 0.595 [0.50,0.69] | **0.653 [0.55,0.76]** | 0.42 [0.24,0.60] | 0.24 [0.08,0.40] | 0.72 [0.52,0.88] | 1.00 |
| −20 | 0.785 [0.70,0.86] | 0.853 [0.77,0.93] | 0.58 [0.40,0.76] | 0.76 [0.60,0.92] | 0.80 [0.64,0.96] | 1.00 |

- By T60 stratum (all samples):
  - δ = 0: [0.2,0.3) 1.00, [0.3,0.4) 1.00, [0.4,0.5] 0.977;
  - δ = +20: 0.72 / 0.46 / 0.57;
  - δ = −20: 0.84 / 0.86 / 0.71;
  - the results for δ = ±10 and all confidence intervals are in the json.

### BLSTM+MVDR (control, no verdict; excluding [0,15))
0.713, 0.673, 0.633, 0.607 and 0.607 at δ = 0, +10, −10, +20 and −20 respectively.

### Verdict (frozen rules)
- acc_±20 = min(0.653, 0.853) = **0.653** < 0.85 → **direction uncertainty is an important bottleneck**.
- [0,15): already down to 0.42–0.58 at δ = ±10; [15,30): down to 0.24 at δ = +20. These two strata are reported separately and take no part in the verdict.
- Note (description only): the results for +δ and −δ are asymmetric, consistent with the sample composition. In the [15,30) stratum, θ2 is on the positive side of θ1 in 19 of the 25 groups ([30,60): 14/25, [60,180]: 9/25), so a +δ query azimuth is more likely to lean towards the interferer.
