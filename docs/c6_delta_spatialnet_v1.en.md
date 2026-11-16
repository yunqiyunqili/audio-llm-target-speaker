English translation of c6_delta_spatialnet_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c6_delta_spatialnet_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C6δ (SpatialNet snapshot) v1 (c6-delta-spatialnet-v1; 2026-11-05; rules written before the run and not changed after the results were seen)

> - Uses only **the epoch-72 snapshot of SpatialNet-small** (`checkpoints/snapshots/spatialnet_ep72_20261105-211918/`, sha256 `582b2a3a…`). This is a **snapshot taken during training, for diagnosis only, and does not represent a completed model**.
> - **δ is a fixed offset artificially added to the true azimuth (simulated error); it is neither an estimated azimuth nor the error of real users**. δ = 0 equals the C5 setting and depends on the oracle azimuth. This document **does not define real C6**.
> - No downloads, no fine-tuning, training jobs not touched, test not used, existing results, prompts and verdicts not changed.

## Inputs and existing outputs
- Front-end output: `data/mix_v1/separated_v1_snapshot/spatialnet_ep72_20261105-211918/` (job 22016833).
- Estimated directions and polarity flags of the two streams: `outputs/frontend_snapshot_eval/spatialnet/selection.jsonl` (v3 rule, srun 22016898). The reference-matched target stream comes from `ref_stream` in `per_sample.jsonl` in the same directory, obtained by PIT against the direct references, **used only for evaluation**.
- Existing downstream outputs of the selected stream: `P2/`, `W/`, `T2/` of job 22016930, **not rerun**.

## Running the other stream
- Other stream = 1 − the selected stream, processed with the v3 polarity flags (both streams of this snapshot are inverted), through the same chain as C5: resample_poly 2/1 → −30 dBFS.
- Run separately with the frozen **P2**, and with **Whisper large-v3 (`06f233fe…`) + the frozen T2**, with settings exactly as in 22016930.
- Implementation: generate the "other stream" selection file with `scripts/sms_wsj/make_other_selection.py` (stream set to 1 − stream, other fields unchanged), then call `scripts/qwen/qwen_omni_snapshot.py` **as is**.
- Job: `scripts/qwen/qwen_omni_snapshot_other.sbatch`, configuration <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes. The working tree must be clean before submission. Record the commit, node, GPU model and per-item time.

## Combination by δ and evaluation (the same rules as c6-delta-v1)
- Query azimuth θ_query = wrap(θ1 + δ), δ ∈ {0, +10, −10, +20, −20}. Stream selection with v3 `choose` on the existing estimated directions of the two streams, stream 0 on a tie. The downstream output of whichever stream is selected is taken.
- At δ = 0 the selection must agree item by item with `selection.jsonl`, and the results at δ = 0 must agree with the evaluation of 22016930; otherwise stop.
- Metrics (`eval_c4_v1.score` and `summ`): stream-selection error rate (with reference matching as the standard), target only, interferer only, neither, target-any, all target values hit, pair-flip accuracy, M, U.
- The two downstream routes (P2, Whisper + T2) are reported separately.
- Angle basis: equal angles (i.e. the samples themselves), and weighted by the natural cv_dev93 angle distribution ([0,15) 78, [15,30) 96, [30,60) 155, [60,180] 653). Pair-flip accuracy is not weighted.
- Strata: four angle strata; three T60 strata ([0.2,0.3), [0.3,0.4), [0.4,0.5]).

## Abstention (the same procedure as c6-delta-v1)
- margin = |wrap(θ_query − θ̂_other)| − |wrap(θ_query − θ̂_selected)|; output "cannot determine" when margin < τ. Uses only the estimated directions, neither reference signals nor speaker identity.
- Grouping: sort the group numbers and shuffle with `random.Random(2026).shuffle`; the first 50 groups are dev, the last 50 eval, exactly the same grouping as for FaSNet.
- τ: on the dev half (five δ pooled), take the smallest τ ∈ {0, 1, …, 90}° making the stream-selection error rate of answered samples ≤ 0.05; if none, take the τ with the smallest error rate.
- On the eval half, for each δ and each downstream route, report: coverage, the stream-selection error rate of answered samples, the proportion of interferer-only, the proportion of target-any, and the no-abstention (τ = 0) control.
- Risk–coverage curves: drawn in `docs/figures/c6_risk_coverage_spatialnet_v1.png`, in the same format as the FaSNet figure.

## Controls
- Reported alongside the FaSNet-TAC C6δ (`docs/c6_delta_v1.md`, P2 only).
- FaSNet-TAC has Whisper + T2 results only at δ = 0 (C7W-C5 = 0.340), not for other δ, because the other stream was never run.
- The gap between P2 and Whisper + T2 is reported for each δ. Descriptive only, no verdict thresholds.

## Implementation
- Evaluation: `scripts/qwen/eval_c6_delta_spatialnet.py`, results in `outputs/c6_delta_spatialnet_v1/`.
- The rules are the same as `eval_c6_delta.py`. The only differences: the source of reference matching is SpatialNet's; both downstream routes are evaluated; and T60 strata are added.

## Results (compiled 2026-11-06; **SpatialNet snapshot during training, for diagnosis only; δ is simulated error, not real C6**)
- Run records:
  - Downstream of the other stream: job 22017085, <NODE>, A100-PCIE-40GB, 2026-11-05 22:32:28–22:44:42, exit code 0, MaxRSS 30.0 GB. The commit recorded by the job is `3ef9ae4`, with no uncommitted dependency files. CPU dry-run srun 22017069. Time per part: P2 295.6 s, Whisper 49.1 s, T2 264.1 s, loading 71.3 s.
  - The selected stream uses the existing outputs of job 22016930.
  - Evaluation: `eval_c6_delta_spatialnet.py`, run on the login node, reading raw outputs only. Results in `outputs/c6_delta_spatialnet_v1/`.
- Consistency: at δ = 0 the selection agrees item by item with `selection.jsonl`; the results of both routes at δ = 0 are exactly identical to the evaluation of 22016930. The script asserts all of this.

### Results for each δ (200 samples, equal angles)
| δ | Stream-selection error rate | Route | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | Natural angle weighting: target-any / interferer-only |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | P2 | 0.530 | 0.000 | 0.470 | **0.635** | 0.530 | 0.33 | 0.633 / 0.000 |
| 0 | | Whisper + T2 | 0.510 | 0.005 | 0.485 | 0.590 | 0.510 | 0.32 | 0.615 / 0.002 |
| +10 | 0.210 | P2 | 0.435 | 0.130 | 0.435 | 0.510 | 0.435 | 0.27 | 0.591 / 0.045 |
| +10 | | Whisper + T2 | 0.430 | 0.100 | 0.470 | 0.480 | 0.430 | 0.28 | 0.577 / 0.034 |
| −10 | 0.130 | P2 | 0.465 | 0.085 | 0.450 | 0.545 | 0.465 | 0.29 | 0.603 / 0.028 |
| −10 | | Whisper + T2 | 0.440 | 0.085 | 0.475 | 0.505 | 0.440 | 0.26 | 0.586 / 0.028 |
| +20 | 0.395 | P2 | 0.335 | **0.245** | 0.420 | 0.385 | 0.335 | 0.20 | 0.530 / 0.100 |
| +20 | | Whisper + T2 | 0.340 | 0.215 | 0.445 | 0.380 | 0.340 | 0.22 | 0.527 / 0.090 |
| −20 | 0.220 | P2 | 0.425 | 0.145 | 0.430 | 0.505 | 0.425 | 0.27 | 0.579 / 0.062 |
| −20 | | Whisper + T2 | 0.395 | 0.135 | 0.470 | 0.460 | 0.395 | 0.24 | 0.560 / 0.058 |

"Both" is 0 for every δ and route.

### By angle stratum (P2; target-any / interferer-only)
| δ | [0,15) | [15,30) | [30,60) | [60,180] |
|---|---|---|---|---|
| 0 | 0.60 / 0 | 0.58 / 0 | 0.74 / 0 | 0.62 / 0 |
| +10 | 0.26 / 0.34 | 0.42 / 0.18 | 0.74 / 0 | 0.62 / 0 |
| −10 | 0.34 / 0.28 | 0.48 / 0.06 | 0.74 / 0 | 0.62 / 0 |
| +20 | 0.26 / 0.34 | 0.12 / 0.46 | 0.54 / 0.18 | 0.62 / 0 |
| −20 | 0.34 / 0.28 | 0.46 / 0.12 | 0.60 / 0.18 | 0.62 / 0 |

Whisper + T2 shows the same stratum trends, e.g. at δ = +20, [15,30) is 0.14 / 0.42, and [60,180] is 0.62 / 0 for every δ. By T60 stratum (P2, [0.2,0.3) / [0.3,0.4) / [0.4,0.5]): at δ = 0, 0.60 / 0.61 / 0.67; at δ = +20, 0.41 / 0.27 / 0.42. Full results are in the json.

### Abstention (using neither reference signals nor speaker identity; dev and eval grouping the same as for FaSNet)
- On dev the rule selects **τ = 17°** (the same as FaSNet): on dev (five δ pooled), coverage 0.538, stream-selection error rate 0.037.
- Eval half (100 samples per δ):

| δ | Coverage | Answered: stream-selection error rate | P2: interferer-only / target-any (without abstention) | Whisper + T2: interferer-only / target-any (without abstention) |
|---|---|---|---|---|
| 0 | 0.72 | 0 | 0 / 0.681 (0 / 0.69) | 0 / 0.639 (0.01 / 0.63) |
| +10 | 0.53 | 0 | 0 / 0.660 (0.15 / 0.56) | 0 / 0.660 (0.10 / 0.53) |
| −10 | 0.72 | 0 | 0 / 0.708 (0.08 / 0.62) | 0 / 0.653 (0.08 / 0.56) |
| +20 | 0.50 | 0.14 | 0.12 / 0.540 (0.31 / 0.39) | 0.08 / 0.540 (0.26 / 0.39) |
| −20 | 0.65 | 0 | 0 / 0.723 (0.09 / 0.61) | 0 / 0.662 (0.09 / 0.55) |

- Risk–coverage curves: `docs/figures/c6_risk_coverage_spatialnet_v1.png`.

### Side by side with the FaSNet-TAC C6δ (P2; target-any, equal angles / natural angle weighting)
| δ | Stream-selection error rate (SpatialNet / FaSNet) | SpatialNet P2 | FaSNet P2 | SpatialNet: P2 − Whisper + T2 |
|---|---|---|---|---|
| 0 | 0.000 / 0.010 | 0.635 / 0.633 | 0.395 / 0.408 | +0.045 |
| +10 | 0.210 / 0.220 | 0.510 / 0.591 | 0.335 / 0.387 | +0.030 |
| −10 | 0.130 / 0.130 | 0.545 / 0.603 | 0.335 / 0.388 | +0.040 |
| +20 | 0.395 / 0.405 | 0.385 / 0.530 | 0.280 / 0.359 | +0.005 |
| −20 | 0.220 / 0.215 | 0.505 / 0.579 | 0.300 / 0.368 | +0.045 |

- FaSNet-TAC has Whisper + T2 results only at δ = 0 (C7W-C5 0.340), not for other δ.

### Description (description only; no verdicts and no significance tests)
- The stream-selection error rates of the two front ends are almost identical (within ±0.01 for every δ), because selection depends only on the estimated directions, and both front ends estimate directions accurately. So the stream-selection problems caused by direction error have little to do with which front end is used.
- For every δ, the end-to-end results of SpatialNet are about 0.1–0.24 higher than FaSNet (equal angles). But at δ = +20, SpatialNet's interferer-only reaches 0.245 (FaSNet 0.14): the cleaner the separation, the more completely the interferer's content is output when the wrong stream is selected.
- On SpatialNet, P2 is 0.005–0.045 above Whisper + T2.
- On the eval half, abstention (τ = 17°) brings the stream-selection error rate and interferer-only down to 0 for all δ except +20, with target-any of answered samples between 0.64 and 0.72; the cost is coverage of only 0.50–0.72.
