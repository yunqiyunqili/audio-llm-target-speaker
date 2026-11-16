English translation of c6_real_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c6_real_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Real C6 v1: the user gives the target direction, the system estimates the source directions (c6-real-v1; **frozen 2026-11-06 11:05 EST**, written before the run and not changed after the results were seen)

> - The draft `c6_real_v1_draft.md` (`ea10147`) was rewritten into this document according to the project lead's decisions of 2026-11-06.
> - Uses the 100 devel groups of mix_v1 (200 swapped samples), **not test**. No GPU jobs are submitted, the other FaSNet-TAC stream is not run retroactively, nothing is downloaded or fine-tuned, and training jobs are not touched; mix_v1, the P2/T2 prompts, slu-score-v2, binding-score-v1, the v3 stream-selection rule and existing results and verdicts are not changed.
> - The main front end is a **snapshot taken during training** of SpatialNet-small at epoch 72; **the results are "snapshot diagnostics"**; FaSNet-TAC is the control. After SpatialNet training completes, rerun with the completed checkpoint under the same rules.
> - All thresholds are empirical, not significance tests.

## 1. Definitions
1. **Real C6 is not defined as "estimating the target direction from the mixture audio"**: the mixture audio itself cannot tell whom the user wants to hear. **The target direction is provided by user input** (query direction θ_query), and **the source directions are estimated by the system from the separated streams** (the `doa` of v3, already in the selection files). Stream selection compares the estimated directions of the two streams with θ_query.
2. User input is simulated in two ways: **C6-sector** (the user only names a sector) and **C6-noise** (the user gives an angle with random error).
3. The fixed offset δ continues to be called **C6δ, a sensitivity analysis only**, not part of real C6; existing results are not rerun, only cited alongside.
4. Azimuth conventions as in `docs/conditions.md`: relative to mic 0, counter-clockwise positive, (−180°, 180°], wrap(a) = (a + 180) mod 360 − 180. θ1 (position 0) is always the target azimuth and θ2 the interferer azimuth; the two swapped samples of a group have the same θ1, θ2.

## 2. Conditions
### 2.1 C6-sector
- Number of sectors K, W = 360°/K.
- **Main partition ("centre")**: the mic 0 direction is the centre of sector 0. k(θ) = ⌊((θ + W/2) mod 360) / W⌋, centre c_k = wrap(k·W).
- **Rotated by half a sector ("rotated", sensitivity analysis only)**: k(θ) = ⌊(θ mod 360) / W⌋, centre c_k = wrap((k + 0.5)·W).
- Query direction θ_query = c_{k(θ1)}.
- **Direction indistinguishable**: k(θ1) = k(θ2) (using only the true azimuths, as an evaluation-side label; defined per group).
- **Main result: K = 8, centre partition.** K = 4, and the rotated partitions for both K, are sensitivity analyses only.

### 2.2 C6-noise
- θ_query = wrap(θ1 + ε), σ ∈ {5°, 10°, 15°}, **10 repetitions** (r = 0, …, 9).
- **Sampled per group**: `rng = np.random.default_rng(np.random.SeedSequence([2026, σ, r]))`, `eps = rng.normal(0, σ, 100)`, assigned in ascending group order (0–99); the two swapped samples of a group share the same ε.
- **Main result: σ = 10°.** σ = 5° and 15° are sensitivity analyses only.
- "Direction indistinguishable" is not defined; results split by the true indistinguishable label of the K = 8 centre partition are reported separately, descriptive only.

## 3. Stream selection, correctness and main basis
- **Stream selection**: the frozen v3 rule (`eval_c6_delta.choose`: take the stream whose estimated direction is closest to θ_query; stream 0 on a tie). The estimated directions and polarity flags of the two streams come from the existing selection files and are not re-estimated.
- **Reference target stream** (evaluation only): for SpatialNet, `ref_stream` of `outputs/frontend_snapshot_eval/spatialnet/per_sample.jsonl`; for FaSNet-TAC, `ref_stream_direct` of `outputs/frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl` (same as C6δ).
- The downstream output of each sample = the existing raw output of the selected stream (see §6); the class is computed with `eval_c4_v1.score` (binding-score-v1, unchanged).
- **Main basis (all 200 samples)**:
  - stream-selection error: selected stream ≠ reference target stream, **or the sample's direction is indistinguishable** (with indistinguishable direction, selection never counts as correct);
  - "target-only", target-any, "all target values hit": indistinguishable samples **count as 0** (no credit);
  - interferer-only: computed from the actual output (indistinguishable samples are counted as they are);
  - pair-flip accuracy: a group counts as correct only if both swapped samples are "target-only" (after crediting).
- **Distinguishable-subset basis**: distinguishable samples only, all metrics computed as they are, with the number of distinguishable samples reported. The **number of indistinguishable samples** (and per angle stratum) and the actual class distribution on indistinguishable samples are also reported separately.
- C6-noise has no indistinguishable samples; its main basis is the actual result over all 200 samples.

## 4. Abstention
- **No abstention**: answer everything.
- **Abstention A (margin)**: margin = |wrap(θ_query − θ̂_other)| − |wrap(θ_query − θ̂_selected)|; **abstain when margin < 17°**. τ = 17° is fixed (selected on the dev half of C6δ) and not tuned further; used for C6-sector and C6-noise and evaluated on all 200 samples (τ = 17° comes from the dev-half groups of C6δ, and those groups are also in this evaluation).
- **Abstention B (sector)**: abstain when the estimated directions θ̂_0 and θ̂_1 of both streams fall inside the query sector k(θ_query). Uses only the estimated directions and the query sector, **no true azimuths, no tuning, C6-sector only** (with the respective K and partition). The 2×2 counts of the abstention-B flag against the true "indistinguishable" label are also reported.
- For each abstention mode (including no abstention), report: **coverage** (proportion answered); among answered samples, the **stream-selection error rate (risk, main basis)**, interferer-only and target-any (credited per the main basis).
- **Risk–coverage curves**: abstention A with τ ∈ {0, 1, …, 90}°, all 200 samples, coverage on the x-axis and the stream-selection error rate of answered samples on the y-axis (unweighted); one subplot per front end, one line per condition (C6-noise uses the mean of the 10 repetitions), with the points for τ = 17° and abstention B marked. The figure is saved as `docs/figures/c6_real_risk_coverage_v1.png`.

## 5. Weighting, confidence intervals and repetitions
- **Equal angles**: plain mean over the 200 samples (50 samples in each of the four strata).
- **Natural angle weighting**: the cv_dev93 angle distribution [0,15) 78, [15,30) 96, [30,60) 155, [60,180] 653 (982 in total). The weight of each sample is v_i = (w_s / 982) / n_s (n_s is the number of samples in that sample's stratum). The weighted proportion on any subset (answered, distinguishable, etc.) = Σ v_i x_i / Σ v_i; weighted coverage = Σ_answered v_i / Σ_all v_i. Pair-flip accuracy is not weighted.
- **95% confidence intervals**: **group-level bootstrap**, stratified by angle stratum (25 groups drawn with replacement per stratum, the two swapped samples of a group together), B = 2000, `np.random.default_rng(2026)`, 2.5% and 97.5% quantiles. Intervals are given for the stream-selection error rate, interferer-only, target-any (equal and weighted), coverage, and for the stream-selection error rate, interferer-only and target-any of answered samples. No interval for pair-flip accuracy.
- **C6-noise**: report the **mean and range (min–max)** over the 10 repetitions; the confidence interval is a bootstrap of the "mean over 10" (after each resample, first average the 10 repetitions, then take quantiles).
- Results stratified by the four angle strata (equal) are also reported.

## 6. Front ends and inputs (existing outputs only)
| Front end / route | Selected stream (v3, θ1) | Other stream |
|---|---|---|
| SpatialNet ep72 snapshot + P2 (main) | `outputs/qwen_omni_snapshot/22016930_spatialnet/P2` | `outputs/qwen_omni_snapshot_other/22017085_spatialnet/P2` |
| SpatialNet ep72 snapshot + Whisper + T2 | `.../22016930_spatialnet/T2` | `.../22017085_spatialnet/T2` |
| FaSNet-TAC + P2 (control) | `outputs/qwen_omni_c5/22015337/C5_fasnet_tac` | `outputs/qwen_omni_c6c7/22015445/OTHER` |
- Selection files: SpatialNet `outputs/frontend_snapshot_eval/spatialnet/selection.jsonl`; FaSNet-TAC `outputs/c5_v1/selection.jsonl`.
- FaSNet-TAC has no Whisper + T2 on the other stream; **it is not run retroactively in this round**, and only P2 is compared.

## 7. Consistency checks (if not satisfied, stop and produce no results)
- With θ_query set to θ1: the stream selection of every sample must agree item by item with the selection file;
- the resulting outcomes (target-only, interferer-only, both, neither, target-any, all target values hit, pair-flip accuracy) must be exactly identical to the stored evaluations: FaSNet-TAC against `C5_fasnet_tac` in `outputs/qwen_omni_c5_eval/22015337/eval_c5_v1.json`; SpatialNet P2 and Whisper + T2 against `outputs/frontend_snapshot_downstream_eval/eval.json` (the evaluation of job 22016930).

## 8. Main-line judgement (written before the run and not changed after the results were seen; empirical)
- **Object**: the two main results of SpatialNet snapshot + P2, judged separately: C6-sector (K = 8, centre partition) and C6-noise (σ = 10°, mean of 10 repetitions). **Point estimates only**, with **natural angle weighting**, main basis.
- Let OI₀ = the weighted interferer-only without abstention. An abstention rule X (A or B for C6-sector; only A for C6-noise) is "effective" if: the weighted interferer-only on answered samples ≤ 0.5 × OI₀, **and** the weighted coverage ≥ 0.60.
  1. OI₀ ≥ 0.05, and at least one abstention rule is effective → **"reducing wrong answers under direction uncertainty" is a viable main line**;
  2. OI₀ < 0.02 → **direction uncertainty is a secondary problem under that condition**;
  3. otherwise → **matters only at small angles or under some conditions**.
- The judgement applies only to this snapshot and these 200 samples; it is a snapshot diagnostic.

## 9. Implementation
- Evaluation: `scripts/qwen/eval_c6_real.py` (importing `choose`, `wrap`, `NATURAL`, `TAUS` from `eval_c6_delta`, and `score`, `summ` from `eval_c4_v1`); results in `outputs/c6_real_v1/`.
- Figure: `scripts/sms_wsj/plot_c6_real_risk_coverage.py` (matplotlib of the tfgridnet environment).
- Run on a CPU node (<CPU_PARTITION>, srun, account <SLURM_ACCOUNT>) with a clean working tree. Read raw outputs only; write nowhere outside `outputs/c6_real_v1/` (except the figure).

## 10. What needs to be run later (not in this round)
1. Whisper + T2 on the other FaSNet-TAC stream (GPU, needs approval).
2. The completed SpatialNet-small checkpoint: front-end inference → cv_dev93 check and mix_v1 stream selection → P2 and Whisper + T2 on both streams → recompute under these rules (GPU, separately approved after training completes).

---

## Results (2026-11-06; the SpatialNet part is a **snapshot diagnostic**)
> The rules above were frozen before the run (`853db98`); below are the results actually obtained under the rules; neither rules nor judgements were changed because of the results. Group-level bootstrap 95% confidence intervals in parentheses; C6-noise is written as "mean [min–max] (interval)", the interval being for the mean over 10 repetitions. "Equal" = plain mean over the 200 samples drawn equally across the four angle strata; "weighted" = weighted by the cv_dev93 natural angle distribution.

### Run record
- srun 22018994 (<CPU_PARTITION>, account <SLURM_ACCOUNT>, 4 CPU, 16 GB, 30 minutes), node <NODE>, 2026-11-06 11:06:26–11:07:28, exit code 0, MaxRSS 0.2 GB. At run time HEAD was `5e91305`, working tree clean.
- Command: `envs/qwen_omni/bin/python -I -B scripts/qwen/eval_c6_real.py outputs/c6_real_v1`, then plotting with the tfgridnet environment. Results: `outputs/c6_real_v1/eval_c6_real.json`, `risk_coverage.json` (not in the repository); figure `docs/figures/c6_real_risk_coverage_v1.png`.
- After plotting, the y-axis title was shortened (the original title was cut off); only the text on the figure was changed and it was redrawn once; the data are unchanged.
- Consistency checks **pass**: with θ_query = θ1, the stream selection of both front ends agrees item by item with the selection files; FaSNet-TAC P2 against the C5 evaluation and SpatialNet P2 and Whisper + T2 against the evaluation of 22016930 are exactly identical on all metrics listed in §7 (script assertion).

### Direction-indistinguishable samples (C6-sector, defined per group, the same for both front ends)
| Sectors | Indistinguishable samples (/200) | [0,15) / [15,30) / [30,60) / [60,180] | Weighted proportion |
|---|---|---|---|
| **K = 8 centre (main)** | **72** | 42 / 24 / 6 / 0 | 0.133 |
| K = 4 centre | 102 | 40 / 42 / 18 / 2 | 0.229 |
| K = 8 rotated | 72 | 40 / 30 / 2 / 0 | 0.129 |
| K = 4 rotated | 120 | 50 / 38 / 30 / 2 | 0.275 |

- Abstention-B flag against the true indistinguishable label (K = 8 centre): for SpatialNet, all 72 indistinguishable samples are abstained by B, plus 2 distinguishable samples abstained; for FaSNet-TAC, 70 / 72 are abstained, 2 indistinguishable samples are answered, and another 2 distinguishable samples are abstained.

### Main result 1: C6-sector, K = 8, centre partition (main basis, all 200 samples)
| Front end / route | Abstention | Coverage equal / weighted | Stream-selection error rate equal / weighted | Interferer-only equal / weighted | target-any equal / weighted |
|---|---|---|---|---|---|
| SpatialNet + P2 (snapshot diagnostic) | none | 1 / 1 | 0.360 (0.290,0.430) / 0.133 (0.105,0.163) | 0.130 (0.075,0.190) / **0.049** (0.028,0.073) | 0.410 (0.345,0.480) / 0.548 (0.460,0.634) |
| | A | 0.590 (0.520,0.650) / 0.846 (0.814,0.876) | 0.034 / 0.009 | 0.025 (0.000,0.065) / 0.007 (0.000,0.018) | 0.602 / 0.622 |
| | B | 0.630 (0.560,0.700) / 0.864 (0.832,0.892) | 0.000 / 0.000 | 0.000 / 0.000 | 0.643 / 0.632 |
| SpatialNet + Whisper + T2 (snapshot diagnostic) | none | 1 / 1 | 0.360 / 0.133 | 0.115 (0.065,0.170) / 0.044 (0.024,0.067) | 0.375 / 0.533 |
| | A | 0.590 / 0.846 | 0.034 / 0.009 | 0.025 / 0.007 | 0.559 / 0.608 |
| | B | 0.630 / 0.864 | 0.000 / 0.000 | 0.000 / 0.000 | 0.587 / 0.614 |
| FaSNet-TAC + P2 | none | 1 / 1 | 0.360 / 0.133 | 0.060 (0.030,0.095) / 0.021 (0.010,0.033) | 0.270 (0.220,0.325) / 0.360 (0.285,0.434) |
| | A | 0.585 / 0.846 | 0.034 / 0.008 | 0.000 / 0.000 | 0.410 / 0.412 |
| | B | 0.640 / 0.867 | 0.016 / 0.004 | 0.008 / 0.002 | 0.422 / 0.416 |

- Pair-flip accuracy (main basis, no abstention): SpatialNet P2 0.22, Whisper + T2 0.21; FaSNet-TAC P2 0.08.
- **Distinguishable subset** (128 samples): the stream-selection error rate of both front ends is 0 (all of SpatialNet's selection errors are on indistinguishable samples); target-any SpatialNet P2 0.641 (weighted 0.632), Whisper + T2 0.586 (0.614), FaSNet-TAC P2 0.422 (0.415); interferer-only SpatialNet 0, FaSNet-TAC 0.008.
- **Actual outputs on indistinguishable samples** (72, no credit, descriptive only): for SpatialNet P2, the proportion selecting the reference target stream is 0.50, interferer-only 0.361, target-any 0.292; for FaSNet-TAC P2, 0.54, 0.153 and 0.194 respectively.
- By stratum (SpatialNet P2, no abstention, equal): interferer-only [0,15) 0.30, [15,30) 0.16, [30,60) 0.06, [60,180] 0; abstention A has coverage 0 in the [0,15) stratum and 0.56 in [15,30) (stream-selection error rate of answered samples 0.143); abstention B has coverage 0.16 in [0,15) and 0.48 in [15,30), and the stream-selection error rate of answered samples is 0 in every stratum.

### Main result 2: C6-noise, σ = 10° (10 repetitions)
| Front end / route | Abstention | Coverage equal / weighted | Stream-selection error rate equal / weighted | Interferer-only equal / weighted | target-any equal / weighted |
|---|---|---|---|---|---|
| SpatialNet + P2 (snapshot diagnostic) | none | 1 / 1 | 0.137 [0.115–0.175] (0.114,0.159) / 0.050 [0.043–0.064] | 0.086 [0.070–0.110] (0.066,0.109) / **0.032** [0.027–0.041] (0.024,0.040) | 0.545 [0.515–0.560] / 0.600 [0.592–0.606] |
| | A | 0.606 [0.560–0.650] / 0.855 [0.829–0.874] | 0.017 / 0.005 | 0.010 [0.000–0.036] / 0.003 [0.000–0.009] | 0.632 / 0.630 |
| SpatialNet + Whisper + T2 (snapshot diagnostic) | none | 1 / 1 | 0.137 / 0.050 | 0.077 / 0.029 | 0.511 / 0.585 |
| | A | 0.606 / 0.855 | 0.017 / 0.005 | 0.009 / 0.003 | 0.596 / 0.618 |
| FaSNet-TAC + P2 | none | 1 / 1 | 0.132 [0.105–0.165] / 0.048 | 0.049 [0.035–0.070] / 0.018 | 0.350 [0.335–0.360] / 0.391 |
| | A | 0.595 / 0.849 | 0.018 / 0.004 | 0.002 / 0.000 | 0.420 / 0.415 |

- By stratum (SpatialNet P2, no abstention, equal, mean of 10): interferer-only [0,15) 0.216, [15,30) 0.094, [30,60) 0.036, [60,180] 0.
- Split by the true indistinguishable label of K = 8 centre (SpatialNet P2): on the 72 indistinguishable samples, stream-selection error rate 0.268 and interferer-only 0.164; on the 128 distinguishable samples, 0.062 and 0.043 respectively.

### Sensitivity analyses (SpatialNet + P2, snapshot diagnostic; main basis; weighted interferer-only / weighted target-any; coverage weighted)
| Condition | No abstention | Abstention A (coverage) | Abstention B (coverage) |
|---|---|---|---|
| sector K = 4 centre | 0.075 / 0.491 | 0.035 / 0.572 (0.837) | 0 / 0.636 (0.771) |
| sector K = 8 rotated | 0.031 / 0.561 | 0.002 / 0.634 (0.859) | 0 / 0.642 (0.870) |
| sector K = 4 rotated | 0.088 / 0.462 | 0.027 / 0.568 (0.789) | 0 / 0.636 (0.723) |
| noise σ = 5° | 0.017 / 0.619 | 0 / 0.635 (0.876) | — |
| noise σ = 15° | 0.039 / 0.594 | 0.006 / 0.626 (0.854) | — |

- The corresponding numbers for FaSNet-TAC + P2, all numbers for Whisper + T2, and the confidence intervals and strata of each condition are in `eval_c6_real.json`.

### Side by side with C6δ (P2, no abstention; C6δ is an existing result, not recomputed)
| Condition | SpatialNet stream-selection error rate / interferer-only (weighted) / target-any (weighted) | FaSNet-TAC, same columns |
|---|---|---|
| C6-sector K = 8 centre | 0.360 / 0.049 / 0.548 | 0.360 / 0.021 / 0.360 |
| C6-noise σ = 10° | 0.137 / 0.032 / 0.600 | 0.132 / 0.018 / 0.391 |
| C6δ = +10° / −10° | 0.21 / 0.13; 0.045 / 0.028; 0.591 / 0.603 | 0.22 / 0.13; 0.026 / 0.018; 0.387 / 0.388 |
| C6δ = +20° / −20° | 0.395 / 0.22; 0.100 / 0.062; 0.530 / 0.579 | 0.405 / 0.215; 0.059 / 0.033; 0.359 / 0.368 |
- Note: the stream-selection error rate of C6-sector counts all indistinguishable samples as wrong, a different basis from the C6δ stream-selection error rate (which compares only with the reference target stream); the actual stream-selection error rate of C6-sector against the reference target stream is SpatialNet 0.180, FaSNet-TAC 0.165.

### Risk–coverage
- Figure: `docs/figures/c6_real_risk_coverage_v1.png` (all 200 samples, equal, main basis; "+" is τ = 17°, "×" is abstention B).
- Without abstention, the stream-selection error rate of C6-sector K = 8 (0.36) comes mainly from indistinguishable samples; at τ = 17° abstention A has coverage of about 0.59 and abstention B about 0.63, and both have stream-selection error rates close to 0 on answered samples.

### Main-line judgement (under the frozen rules of §8; SpatialNet + P2; weighted point estimates)
| Main result | OI₀ | Abstention A: interferer-only / coverage / effective | Abstention B: interferer-only / coverage / effective | Judgement |
|---|---|---|---|---|
| C6-sector K = 8 centre | **0.0489** | 0.0069 / 0.846 / yes | 0.0000 / 0.864 / yes | **matters only at small angles or under some conditions** |
| C6-noise σ = 10° | **0.0320** | 0.0028 / 0.855 / yes | — | **matters only at small angles or under some conditions** |
- The abstention rules of both main results are "effective", but OI₀ is below 0.05 for both, so by the rules they fall into category 3. The OI₀ of C6-sector = 0.0489 is only 0.0011 below 0.05, and its confidence interval (0.028,0.073) crosses 0.05; the judgement uses point estimates only under the frozen rules and is not changed because of this.
- This judgement applies only to the ep72 snapshot, these 200 samples and one downstream run; it is a snapshot diagnostic.
