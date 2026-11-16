English translation of frontend_mix_v1_eval_v2.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：frontend_mix_v1_eval_v2.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# frontend_mix_v1 evaluation v2: adding direct-sound references (frontend-mix-v1-eval-v2; 2026-11-05; content and verdicts written before computation and not changed after the results were seen)

## Scope
- **Only re-evaluates** the existing `data/mix_v1/separated_v1/{blstm_mvdr,fasnet_tac}/` outputs (job 22014183). The front ends are not rerun, Qwen is not run, and C4/C5 are not run.
- mix_v1, existing results and the v1 verdict (`docs/frontend_mix_v1_plan.md`) are not changed. The v1 results and verdict stay as they are.
- No direction matching v2 is written: the direction-estimation method is exactly the same as v1 (`eval_frontend_mix_v1.py: doa`), still taking stream 0 when both directions are equal.

## Direct-sound references (the same definition as in front-end training and SMS-WSJ evaluation)
- The definition comes from `../speech_separation/multichannel/data/prepare.py`:
  - hd = `rir_generator.generate(c=343, fs=8000, r=mic0, s=声源位置, L=房间尺寸, reverberation_time=0, nsample=8192, order=0, mtype=omnidirectional)`;
  - y = `extract_piece(fftconvolve(源信号, hd), offset − rir_start(完整 6 通道 RIR), T)` × gain.
- Applied to mix_v1: the geometry takes mic 0 of the group's cv_dev93 sample, the position of the source (`source_position[:, pos]`) and the room dimensions; the source signal is SLURP downsampled to 8 kHz (`resample_poly` 1/2); offset uses the mix_v1 offset; rir_start uses the value recorded in the mix_v1 manifest; **gain uses the gain recorded in the mix_v1 manifest**. The direct-sound references therefore use the same gains and the same alignment as the early and full images.
- Environment: rir_generator 0.3.0 is installed only in the previously used `../speech_separation/envs/sms_wsj`, used read-only and not modified. The tfgridnet environment does not have this package, and it is not installed.
- Output: the direct-sound references are written to `outputs/frontend_mix_v1_eval_v2/direct_refs/` (npz format, mic 0), not to `data/`.

## Validation on cv_dev93 first
- Draw 50 groups with `random.Random(2026).sample(sorted(cv_dev93 ids), 50)` (the same as B and C of frontend_sanity_v1), recompute the direct sound with the definition above (official gain = 10^(log_weight/20) / std(full 6-channel image) / 71), and compare with `sms_wsj_direct_v1/cv_dev93/<id>_<k>.wav` used by the official evaluation.
- **Pass condition**: the maximum absolute difference over the 100 references < 1e-6 (the official files are float32). If it fails, stop and do not run the later evaluation.

## Metrics
- For both front ends, report the two-talker mean SI-SDR and SI-SDRi, and the target-stream SI-SDR and SI-SDRi, under three references: direct, early and full image (the algorithms for early and full image are the same as v1). SI-SDR uses the original `tfgridnet/objectives.py: sisdr`.
- Like-for-like comparisons:
  - the original records on SMS-WSJ test_eval92 (direct-sound reference): SI-SDRi 4.412 for BLSTM+MVDR, 14.895 for FaSNet-TAC;
  - the results of frontend_sanity_v1 on 50 cv_dev93 groups (direct-sound reference): 4.17 and 15.05.

## Reference matching
- **Switch to the direct references** for PIT; the stream corresponding to the target reference is the target stream.
- Also report its agreement rate with early-reference matching (v1).

## Direction matching and the C5 verdict (threshold unchanged)
- Direction-matched target stream: the same method as v1, taking the stream minimising |wrap(θ̂_k − θ1)|, stream 0 on equality.
- **Direction-matching accuracy (v2)** = the proportion in which the direction-matched target stream agrees with the **direct-reference-matched** target stream.
- Computed only on the 150 samples whose angle is not in [0,15):
  - accuracy ≥ 0.85: the front end **may proceed to C5**;
  - otherwise **it does not proceed to C5**, and the failed samples are listed.
- [0,15) is only reported and takes no part in the verdict. The threshold is empirical.
- Also reported: the proportion of samples where the estimated directions of both streams are exactly the same, and where they differ by ≤ 5° (all 200 samples, and the 150 excluding [0,15)).

## Directions of the two FaSNet-TAC streams
- Using direct-reference matching, the two streams are assigned as stream A and stream B; report for each the proportion whose estimated direction is closer to θ1, with 95% confidence intervals (group-unit bootstrap, 10,000 times, `default_rng(2026)`).
- Reported by angle, T60, active overlap ratio and full-image SIR strata, with the same stratum boundaries as v1. The same is reported for BLSTM+MVDR, as a control.

## Implementation
- First step (sms_wsj environment, CPU, srun): `scripts/sms_wsj/direct_refs_mix_v1.py`, first the cv_dev93 validation, then the direct-sound references of mix_v1.
- Second step (tfgridnet environment, CPU, srun): `scripts/sms_wsj/eval_frontend_mix_v1_v2.py`.
- Results: `outputs/frontend_mix_v1_eval_v2/`.

## Results (2026-11-05; re-evaluation of existing outputs only)
- Run records:
  - First step: srun 22014507 (<CPU_PARTITION>, sms_wsj environment), script commit `2b5264b`.
  - Second step: srun 22014513 (<CPU_PARTITION>, tfgridnet environment).
- Results in `outputs/frontend_mix_v1_eval_v2/`: `direct_refs/` (npz, about 66 MB, not committed), `direct_refs_summary.json`, `eval_v2.json`, `eval_v2_per_sample.jsonl`.
- **cv_dev93 validation passes**: the 100 direct-sound references of the 50 groups differ from the official `sms_wsj_direct_v1` by at most 1.44e-8 in absolute value (required < 1e-6).
- The early image recomputed by the same method on mix_v1 differs from the written file by at most 2.79e-8.
- The direction estimates agree with v1: the v2 script recomputes doa once per sample and asserts that the results are identical.

### Results under the three references (mean over all 200 samples; SI-SDRi = output − observation mic 0)
| Front end | Reference | SI-SDR of observation mic 0 | Two-talker mean SI-SDR | Two-talker mean SI-SDRi | Target-stream SI-SDRi |
|---|---|---|---|---|---|
| BLSTM+MVDR | direct | −5.62 | −3.90 | **1.72** | 1.72 |
| BLSTM+MVDR | early | −0.93 | 1.97 | 2.90 | 2.89 |
| BLSTM+MVDR | full | −0.04 | 2.89 | 2.92 | 2.93 |
| FaSNet-TAC | direct | −5.62 | 5.76 | **11.38** | 11.42 |
| FaSNet-TAC | early | −0.93 | −0.45 | 0.47 | 0.36 |
| FaSNet-TAC | full | −0.04 | −1.09 | −1.05 | −1.16 |

### Like-for-like comparison with SMS-WSJ (direct-sound reference, two-talker mean SI-SDRi)
| Front end | mix_v1 | test_eval92 (original record) | Gap | 50 cv_dev93 groups (frontend_sanity_v1) | Gap |
|---|---|---|---|---|---|
| BLSTM+MVDR | 1.72 | 4.41 | −2.69 | 4.17 | −2.45 |
| FaSNet-TAC | 11.38 | 14.90 | −3.51 | 15.05 | −3.67 |

### Reference matching and direction matching
| | BLSTM+MVDR | FaSNet-TAC |
|---|---|---|
| Agreement between direct-reference and early-reference matching (all 200 / the 150 excluding [0,15)) | 0.96 / 0.973 | 0.995 / 1.00 |
| Direction-matching accuracy v2 (all 200) | 0.735 | 0.500 |
| **Direction-matching accuracy v2 (excluding [0,15), 150)** | **0.713** | **0.513** |
| **Verdict under v2** | **does not proceed to C5** | **does not proceed to C5** |
| Failed samples (excluding [0,15)) | 43 | 73 |
| Both streams exactly the same direction (all 200 / 150) | 0.335 / 0.347 | 0 / 0 |
| Both streams within ≤ 5° (all 200 / 150) | 0.805 / 0.767 | 0.005 / 0.007 |
| Median direction error target stream / interferer stream (all 200) | 2.9° / 2.3° | 77.9° / 2.0° |

- Direction-matching accuracy by angle stratum ([0,15), [15,30), [30,60), [60,180]):
  - BLSTM+MVDR: 0.80 / 0.78 / 0.70 / 0.66;
  - FaSNet-TAC: 0.46 / 0.48 / 0.30 / 0.76.
- Direct SI-SDRi of FaSNet-TAC by angle stratum: 8.41 / 11.59 / 12.56 / 12.98.
- The results by T60, active overlap ratio and full-image SIR strata, and the complete list of failed samples, are all in `eval_v2.json`.

### Proportion of each FaSNet-TAC stream whose estimated direction is closer to θ1 (streams assigned as A and B by direct-reference matching; group-unit bootstrap 95% confidence intervals)
- All 200 samples: stream A 0.41 [0.34, 0.48], stream B 0.40 [0.33, 0.47]. BLSTM+MVDR as control: 0.475 [0.43, 0.52] and 0.46 [0.41, 0.51].
- By angle stratum (stream A / stream B): [0,15) 0.42 / 0.32, [15,30) 0.48 / 0.46, [30,60) 0.30 / 0.34, [60,180] 0.44 / 0.46.
- By T60 stratum: [0.2,0.3) 0.43 / 0.44, [0.3,0.4) 0.27 / 0.23, [0.4,0.5] 0.47 / 0.44.
- By active overlap ratio stratum: [0,0.5) 0.67 / 0.67 (only 9 samples), [0.5,0.75) 0.44 / 0.38, [0.75,0.9) 0.35 / 0.35, [0.9,1] 0.43 / 0.50.
- By full-image SIR stratum: < −0.5 dB 0.48 / 0.48, [−0.5, 0.5] dB 0.41 / 0.41, > 0.5 dB 0.37 / 0.22.

### Summary (description only)
- Evaluated with the direct references, FaSNet-TAC has an SI-SDRi of 11.4 dB on mix_v1, about 3.5 dB lower than on SMS-WSJ under the same basis; BLSTM+MVDR has 1.7 dB, about 2.6 dB lower than on SMS-WSJ under the same basis.
- Neither front end proceeds to C5 under v2:
  - BLSTM+MVDR: the two stream directions are often the same or very close (77% within ≤ 5°);
  - FaSNet-TAC: separation is good, but the direction-estimation error of the target stream is large (median 77.9°), whereas for the interferer stream it is only 2.0°. The cause has not yet been found.
- The v1 results and verdict remain unchanged.
