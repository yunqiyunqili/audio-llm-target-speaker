English translation of frontend_mix_v1_plan.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：frontend_mix_v1_plan.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Zero-shot inference and evaluation of separation front ends on mix_v1 (frontend-mix-v1; 2026-11-05; method and verdict rules written before the run and not changed after the results were seen)

## Background
- The result of c3_gate_v1 is "stop, insufficient evidence"; it **did not pass**. Based on the factor diagnosis (main source two-talker overlap, see `docs/c3_factor_diag_v1.md`), the project lead decided to proceed to the separation front end.
- The leakage check is not yet complete; reporting by full-image SIR stratum continues.
- This stage **does not run Qwen**; c3_gate_v1, mix_v1, the P2 prompt, binding-score-v1 and existing results are not changed; pair flips are not computed.

## Front ends (zero-shot, checkpoints read-only)
| Front end | Code | Checkpoint (read-only) |
|---|---|---|
| BLSTM+MVDR | `../speech_separation/multichannel/blstm_mvdr` (`Model` and `normalize` from the original `train.py`) | `exp/sms_blstm_mvdr_v1/best.pt` |
| FaSNet-TAC | `../speech_separation/multichannel/fasnet_tac` | `exp/sms_fasnet_tac_v1/best.pt` |

- Both models were trained on SMS-WSJ with the mic 0 direct sound as the training target. This time they are used **zero-shot**, without training or fine-tuning.
- Checks before loading: the checkpoint sha256 matches the original record (`test_summary.json`); `package['config']` matches `run_config.json`.
- Inference as in the original `evaluate()`: x = observation / std(observation), output pred = model(x) × std. The output is two single-channel estimates at mic 0, shape (2, T). FP32, batch 1, random seeds and determinism settings as in the original code.
- Environment: `<CONDA_ROOT>/envs/tfgridnet`, unchanged. Job configuration: <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 32 GB, 1 hour. No `__pycache__` written into `../`.

## Input and output
- Input: the 200 `data/mix_v1/devel_seed2026/gNNN_sS/obs_6ch_8k.wav` of mix_v1 (job 22013065).
- Output: `<SHARED_ROOT>/data/mix_v1/separated_v1/<frontend>/gNNN_sS/out0.wav` and `out1.wav` (8 kHz, single channel, float32), plus a `manifest.jsonl`. If the directory already exists, raise an error; **existing files are not overwritten**.
- Expected disk usage about 70 MB (2 front ends × 200 samples × 2 streams × mean about 2.6 s × 8000 × 4 B). If above 2 GB, stop and report.

## Reference signals
- **early image**: the `early_{A,B}_6ch_8k.wav` already written by mix_v1, taking mic 0.
- **full image**: mix_v1 did not write this file. The mic 0 full image is recomputed in memory with the same functions as `scripts/sms_wsj/mix_slurp_v1.py` and the gains in the manifest, without writing files. Check: recompute the early image in the same way and compare with the written file; the maximum absolute difference must be < 1e-6 (float32 precision), otherwise stop.
- Both front ends use exactly the same reference signals.

## Metrics
- SI-SDR: the same as the original `tfgridnet/objectives.py: sisdr` (zero-mean, scale-invariant).
- For each sample, PIT between the two output streams and the two references [target, interferer], taking the permutation with the larger two-talker mean SI-SDR (the same as the original `scores()`).
- Reported:
  - two-talker mean SI-SDR, and the SI-SDR of the target stream;
  - SI-SDRi = SI-SDR − SI-SDR(observation mic 0, reference);
  - computed separately for the early and full-image references.
- Strata (exploration only): angle [0,15), [15,30), [30,60), [60,180]; T60 [0.2,0.3), [0.3,0.4), [0.4,0.5]; active-frame overlap ratio [0,0.5), [0.5,0.75), [0.75,0.9), [0.9,1]; full-image SIR < −0.5, [−0.5, 0.5], > 0.5 dB. Overlap ratio and SIR are taken from the manifest of the generation job.

## Reference matching (C4, offline diagnosis only)
- PIT with the early-image references; the stream corresponding to the target reference is the "reference-matched target stream". The target stream selected with the full-image references, and whether the two agree, are also reported.

## Direction matching (using neither reference signals nor speaker identity)
Using only the output streams y_k, the 6-mic observations x_m and the array geometry (microphone coordinates in the cv_dev93 sample, `sms_wsj.json`):
1. STFT: Hann window, 512 points, hop 128 (8 kHz). Only bins in 80–3900 Hz are used.
2. For each stream k and each microphone m = 1…5: G_{k,m}(f) = Σ_t conj(Y_k(t,f)) · X_m(t,f), then PHAT normalisation: Ĝ = G / max(|G|, 1e-12).
3. Far-field plane wave, horizontal plane only (the z coordinates of the points are similar, so height is ignored): for room azimuth φ (0–359°, step 1°), τ_m(φ) = −((p_m − p_0)·u(φ)) / c, where u(φ) = (cos φ, sin φ), c = 343 m/s, and p are the x-y coordinates of the microphones.
4. Score S_k(φ) = Σ_m Re Σ_f Ĝ_{k,m}(f) · exp(j2πf τ_m(φ)); estimated direction φ̂_k = argmax S_k.
5. Convert to azimuth relative to mic 0: θ̂_k = wrap(φ̂_k − az(mic0 − array centre)), the same as the definition of az_rel (`scripts/sms_wsj/sms_wsj_geometry_stats.py`).
6. **Direction-matched target stream** = the stream minimising |wrap(θ̂_k − θ1)|. If both streams are equally close, take k = 0.
- Check of the method itself (diagnosis only; uses reference signals; takes no part in any matching or verdict): substitute the reference early image as y_k into the same flow, and report the errors of the estimated directions against θ1 and θ2.

## Verdict (judged separately for each front end; thresholds are empirical)
- **Direction-matching accuracy** = the proportion in which the direction-matched target stream agrees with the reference-matched (early image) target stream.
- Computed only on samples whose angle is **not in [0,15)** (150):
  - accuracy ≥ 0.85: the front end **may proceed to C5**;
  - accuracy < 0.85: the front end **does not proceed to C5 for now, only C4 is done**, and the failed samples are listed.
- The [0,15) stratum is only reported and takes no part in the verdict.
- Also reported: the single-stream direction-estimation errors of FaSNet-TAC and BLSTM+MVDR (θ̂ of the reference-matched target stream against θ1, θ̂ of the interferer stream against θ2), and the accuracy by the strata above.
- The two swapped samples of each group are both kept and computed separately; pair flips are not computed.

## Implementation
- Inference: `scripts/sms_wsj/frontend_mix_v1.py` and `.sbatch` (tfgridnet environment).
- Evaluation: `scripts/sms_wsj/eval_frontend_mix_v1.py` (tfgridnet environment, CPU, run with srun).

## Results (job 22014183; **pending a pipeline check; no conclusion about front-end performance for now**)
> The numbers below were obtained by actually running this plan, but the inference pipeline, mixing pipeline and the effect of duration have not yet been checked (see `docs/frontend_sanity_v1.md`). Until the check is complete, **no conclusion about front-end performance is drawn from them**.

- Inference: job 22014183, <NODE>, 2026-11-05 12:44:22–12:46:10, exit code 0; commit `944de58`, no uncommitted files at run time; CPU dry-run srun 22014178. The sha256 and configuration of both checkpoints match the original records (BLSTM+MVDR epoch 62, FaSNet-TAC epoch 98).
- Output: `data/mix_v1/separated_v1/{blstm_mvdr,fasnet_tac}/`, 33 MB each.
- Evaluation: `scripts/sms_wsj/eval_frontend_mix_v1.py` (commit `bd2d1b0`), srun 22014185, results in `outputs/frontend_mix_v1_eval/22014183/`.
- The full-image references were recomputed in memory. Check: the early image recomputed by the same method differs from the written file by at most 2.8e-8.
- Check of the direction estimator: with the reference early image substituted, the median errors are 0.5° for the target and 0.6° for the interferer, and all sample errors are ≤ 10°.

| | Reference | Two-talker mean SI-SDR | Two-talker mean SI-SDRi | Target-stream SI-SDR | Target-stream SI-SDRi |
|---|---|---|---|---|---|
| mixture signal mic 0 (baseline) | early / full | −0.93 / −0.04 | — | — | — |
| BLSTM+MVDR | early | 1.97 | 2.90 | 1.98 | 2.89 |
| BLSTM+MVDR | full | 2.89 | 2.92 | 2.90 | 2.93 |
| FaSNet-TAC | early | −0.45 | 0.47 | −0.56 | 0.36 |
| FaSNet-TAC | full | −1.09 | −1.05 | −1.19 | −1.16 |

| | BLSTM+MVDR | FaSNet-TAC |
|---|---|---|
| Target stream agrees between early and full references | 0.99 | 1.00 |
| Direction-matching accuracy (all 200 samples) | 0.725 | 0.495 |
| Direction-matching accuracy (excluding [0,15), 150 samples) | 0.713 | 0.513 |
| Verdict under the v1 rule | does not proceed to C5, C4 only | does not proceed to C5, C4 only |
| Failed samples (excluding [0,15)) | 43 | 73 |
| Median direction error target stream / interferer stream | 3.1° / 2.3° | 77.9° / 1.9° |
| Samples with one stream nearly silent | 0 | 4 |

- Direction-matching accuracy by angle stratum: BLSTM+MVDR 0.76 / 0.82 / 0.68 / 0.64; FaSNet-TAC 0.44 / 0.48 / 0.30 / 0.76 ([0,15), [15,30), [30,60), [60,180]).
- Among the 150 BLSTM+MVDR samples excluding [0,15):
  - in 52 samples the estimated directions of the two streams are exactly the same, so stream 0 is taken by the rule; accuracy in this part 0.48;
  - in 115 samples the two directions differ by ≤ 5°;
  - in the 35 samples where the two directions differ by > 5°, accuracy is 0.97.
- The v1 verdict is kept unchanged and not revised.
