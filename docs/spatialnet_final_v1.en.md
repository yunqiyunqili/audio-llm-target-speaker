English translation of spatialnet_final_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：spatialnet_final_v1.md

# SpatialNet final checkpoint evaluation v1

Follows the approved configurations and frozen rules of `lora_v2_runs.md` §20–22, `frontend_compare_v1.md` and `c6_real_v1.md`. Only devel inference and evaluation; the final SLURP test is not used, and the original checkpoint, snapshot results and existing verdicts are not changed.

## Confirmation of training completion (2026-11-07 22:33 EST)
- Training chain 21977586 → 21988892 → 21995307 → 22010857 → 22025125 all COMPLETED / ExitCode 0:0; the last job ended at 22:33:10, with no new continuation and no running/queued job of the same name.
- Epochs 1–100 of `metrics.jsonl` are contiguous and complete, agreeing line by line with the EPOCH lines of the segment logs.
- `latest.pt`: epoch100, complete=True, history consistent with the complete metrics.
- `best.pt`: ep100, minimum valid_loss over the complete history = −20.314303703560604, dev SI-SDR = 20.31430916242347 dB; train_loss = −16.48988272721007. best happens to be the last epoch, but the criterion still does not require best.complete=True.
- best/latest load completely, and the configuration agrees with run_config.json after a JSON round trip. The completed_epoch, selected_epoch, checkpoint SHA256, audit SHA256, target and channel configuration of the existing SMS-WSJ test_summary all match this training. Only the source of the summary produced by the original training script was checked; no test was newly run.
- Completion-check script: `scripts/sms_wsj/spatialnet_final_check.py`. Source run directory `<SEPARATION_REPO>/multichannel/spatialnet/exp/sms_spatialnet_small_v1/`.

## Copy of the final model
- `<SHARED_ROOT>/checkpoints/final/spatialnet_ep100_20261107-223434/best.pt`
- 14,622,162 bytes; SHA256 `21caf0fc3b08cfc3b9860efbfeff9a756acecbe36101620378e47891edd76c10`.
- The size and mtime_ns of the source file are the same before and after copying, and the size and SHA256 of the copy match the source and the original training summary. The `snapshot.json` next to it is a provenance file compatible with the existing inference scripts, explicitly marked FINAL, recording the completion check, the job chain and the before/after metadata of the source file.

## Approved execution configuration (written before the results of this final inference)
1. `scripts/sms_wsj/frontend_final.sbatch`: <SLURM_ACCOUNT> / <GPU_PARTITION>, 1 A100, 4 CPU, 32 GB, 20 minutes. Calls the original `frontend_snapshot.py --mode cv` and `--mode mix` in order; only adds a `--final` provenance tag and a completion-metadata check; inference, shape checks and scoring rules unchanged. If the 50-group cv_dev93 check fails, mix is not run. Stop on output errors, wrong shapes or mean direct SI-SDRi ≤0; no automatic retry on failure.
2. cv check results go to `outputs/frontend_final/<job>/spatialnet/`; separated audio goes to `<SHARED_ROOT>/data/mix_v1/separated_final/spatialnet_ep100_20261107-223434/`, a new directory, without overwriting the snapshot.
3. After separation completes and the files pass checks, generate v3 stream selection and reference matching with the original `eval_snapshot_mix.py` on <CPU_PARTITION>. The subsequent P2 / Whisper+T2 on both streams use the approved 30-minute exception; other new jobs 20 minutes.
4. C5, C6δ and c6-real-v1 use the final paths, with scoring, stream selection, noise seeds, abstention rules and consistency checks kept at their frozen versions. New results are placed alongside the ep72 snapshot, in separate output directories.
5. On the same final separated audio, complete the Audio LoRA C5/c6-real-v1 checks, distinguishing entity hits by whether the actually selected stream is the reference target stream or the interferer stream per §11; the sector-indistinguishable penalty is not conflated with actual routing errors. The C3 failure and the fixed 1e-4 step440 are unchanged; Whisper+T2-LoRA is on hold.

Before this start, shared storage usage was 88%, with about 127 GB available; the copy is about 14 MB and the separated audio is expected to be about 33 MB; no download or new environment needed.

## Job record
- Front-end job **22029593** submitted and started running, node <NODE>, A100-PCIE-40GB. Run commit `1815eb5`, the log confirms DIRTY 0; log `outputs/logs/frontend_final_22029593.out`. Completion of the cv check and mix output is still to be verified; no pass is pre-judged.

## Front end and stream selection completed (2026-11-07)
- 22029593 COMPLETED / 0:0, 1:32. cv_dev93 50 groups, errors=[], mean direct SI-SDRi 25.93294 dB, SI-SDR 20.32118 dB; direction matching 0.32 before polarity correction, 1.00 after (also 1.00 excluding angles below 15°), PASS. The polarity-correction rate is 1.00 for both streams. mix log MIX_DONE 200.
- CPU stream-selection job 22029595 (<SLURM_ACCOUNT>/<CPU_PARTITION>, 4 CPU, 16 GB, 20 minutes), COMPLETED / 0:0, 1:05. The 400 files pass the checks of 8kHz, length, single-channel shape, finite values and the set of 200 samples; stream selection generated by calling the original v3 functions; outputs `outputs/frontend_final_eval/ep100_20261107/spatialnet/{selection.jsonl,other_selection.jsonl,per_sample.jsonl}`.
- mix_v1 mean two-talker direct SI-SDRi 18.62176 dB, target stream 18.69140 dB; stream selection 199/200=0.995, 1.00 excluding [0,15); the only error is in [0,15). The stream-selection rule is not changed, and the model is not changed because of the results.
- Side by side with ep72: mean two-talker SI-SDRi 18.55517 → 18.62176 dB; target stream 18.58684 → 18.69140 dB; overall stream selection 1.00 → 0.995, 1.00 for both excluding [0,15). The only error is `g038_s1` (ref_stream=1, sel=0, estimated directions of the two streams no more than 5° apart). This is not a downstream C5/C6 result.

## Downstream entry points (written before the final downstream results)
- `qwen_omni_final.sbatch`: <SLURM_ACCOUNT>/<GPU_PARTITION>, 1 A100, 4 CPU, 48GB, 30 minutes; one job each for the selected and the other stream, calling the original `qwen_omni_snapshot.py`, output `outputs/qwen_omni_final/<job>_<route>/`. The new parameters only mark final and provide an optional branch for a fixed LoRA adapter; baseline generation parameters, P2/T2, Whisper and audio processing are unchanged.
- The two-stream Audio LoRA runs are submitted after the baseline and final-evaluation checks: `eval_audio_lora_final.sbatch`, likewise 1 A100, 4 CPU, 48GB, 20 minutes, P2 only; fixed `step1/lr1e-4/ckpt_step440`, loaded in the same way as the existing step 1 scripts. Reuses the same audio-processing/decoding functions as the baseline; saves the SHA256 of the processed audio and the prompts, which must agree item by item with the same-stream baseline before evaluation.
- `eval_final_routes.py` only wires up the final input paths, calls the original C5/C6δ/c6-real functions and keeps the consistency assertions. C6δ keeps the dev-half τ-selection procedure of its original document (sensitivity analysis only); real C6 remains fixed at margin τ=17° with the K=8/σ=10° main conditions; re-tuning is prohibited.

## Anomaly in the CPU pre-check (2026-11-07, not retried)
- Regression check 22029603 (<SLURM_ACCOUNT>/<CPU_PARTITION>, 4 CPU, 16 GB, 20 minutes), testing the new-path entry points with the existing ep72 outputs, FAILED / 1:0, ran for 1 s. The error was `ModuleNotFoundError: eval_c6_real`: the new entry point did not explicitly add its own directory under `python -I` isolated mode. Scoring did not start, there were no new model predictions, and no original result was overwritten.
- A minimal fix was prepared: only add `Path(__file__).resolve().parent` to `sys.path` before the entry point's imports, consistent with the project's existing evaluation scripts. Experimental configuration, scoring and frozen rules are unchanged. The regression job was not resubmitted; awaiting the project lead's approval to rerun with the original CPU configuration. The final two-stream baseline GPU and LoRA GPU jobs have not yet been submitted.
- Independent input dry-run 22029602 (same CPU configuration) COMPLETED / 0:0, 2:56. The baseline selected stream, baseline other stream and LoRA selected stream each constructed all 200 audio items and tested the processor inputs of the first 2; three DRY_RUN_PASS. The processed-audio SHA256 of the baseline selected stream and LoRA agree 200/200 item by item. No model predictions were produced.

## Regression check rerun with approval and passed (2026-11-07)
- With the project lead's approval, using fix commit `5022bff`, CPU job **22029635** kept 4 CPU / 16 GB / 20 minutes, COMPLETED / 0:0, 35 s. Separate directory `outputs/final_eval_regression/ep72_retry_5022bff/`.
- The P2/T2/Whisper input-set checks for the original 200 samples of each stream pass; C5 scoring, C6δ (except descriptive text), all SpatialNet conditions of c6-real with their consistency checks, and the risk–coverage curves are all exactly identical, item by item, to the original ep72 results. The five output files and their SHA256 are recorded in `regression_check.json`, passed=true.
- Baseline GPU jobs **22029639** (selected) and **22029640** (other) submitted with the originally approved configuration, with a clean working tree at submission, commit5022bff. Completion not yet judged.

## Implementation notes for the final paired analysis (before the LoRA final inference results)
- `scripts/lora/eval_final_lora_checks.py` reuses the original score/summ and the real-C6 stream-selection/sector/noise rules, performing the established angle-stratified paired group bootstrap with B=2000, seed2026. The two azimuth swaps and the 10 noise repetitions of a group are all resampled together; the selection half/report half keep the existing 50-group split. The C6 primary metrics and abstention denominators are asserted item by item to agree with the frozen evaluation outputs.
- For the same stream, the processed-audio SHA256, prompts, sample labels and stream of the baseline and LoRA must be identical item by item; the step440 adapter is fixed. First, using the same ep72 inputs for both sides, it is verified that all paired differences and intervals are zero.
- §11 groups by whether the actually selected stream equals the reference target stream, reporting total/mean predicted entities, numbers of distinguishable target/interferer values hit, and the four binding-class counts; for noise conditions the counts use the number of observations over the 10 repetitions as the denominator. The sector-indistinguishable penalty is kept separately in the primary metric and not conflated with actual stream-selection errors.
- The gates still use all 200 samples: C5 pair flip not below P2 on the same input, and naturally weighted interferer-only under the real C6 main conditions not above P2+0.01. The supplementary diagnostics do not change the primary scoring, the gates or the earlier C3 failure conclusion.

## Final baseline downstream completed (2026-11-07)
- 22029639 selected: COMPLETED/0:0, 11:49; 22029640 other: COMPLETED/0:0, 10:40. Each P2/Whisper/T2 has 200 items, unique keys complete, ALL_DONE. Run code 5022bff/7bc26e5 respectively; the inference and scoring dependencies used are unmodified, and the provenance records show no dirty files.
- CPU final scoring 22029669: COMPLETED/0:0, 39 s; all true-azimuth consistency assertions of P2/WT2 pass. New outputs: `outputs/c5_final/ep100_20261107/eval.json`, `outputs/c6_delta_final/ep100_20261107/`, `outputs/c6_real_final/ep100_20261107/`.
- Self-test of the paired diagnostics 22029646: COMPLETED/0:0, 4 s; with the same ep72 inputs all differences and intervals are zero, and the primary C6 metrics agree with the frozen outputs. Self-test output `outputs/final_eval_regression/ep72_lora_checks_selftest/checks.json`.

| C5 metric | ep72 P2 | final P2 | ep72 Whisper+T2 | final Whisper+T2 |
|---|---:|---:|---:|---:|
| target_any | 0.635 | 0.620 | 0.590 | 0.570 |
| only_interferer | 0.000 | 0.000 | 0.005 | 0.005 |
| pair_flip_accuracy | 0.330 | 0.300 | 0.320 | 0.300 |

Real C6 main conditions (natural angle weighting; with abstention, error rates use answered samples as the denominator; the sector main target hit includes the indistinguishable penalty):

| Condition/strategy | Route | Target hit ep72→final | Interferer-only ep72→final | final stream-selection error | final coverage |
|---|---|---:|---:|---:|---:|
| sector_K8_center/none | P2 | 0.54788 → 0.52519 | 0.04894 → 0.05051 | 0.13454 | 1.00000 |
| sector_K8_center/A | P2 | 0.62239 → 0.59558 | 0.00693 → 0.00924 | 0.01156 | 0.84599 |
| sector_K8_center/B | P2 | 0.63222 → 0.60732 | 0.00000 → 0.00000 | 0.00000 | 0.86155 |
| sector_K8_center/none | WT2 | 0.53253 → 0.51411 | 0.04418 → 0.04733 | 0.13454 | 1.00000 |
| sector_K8_center/A | WT2 | 0.60800 → 0.58623 | 0.00693 → 0.00924 | 0.01156 | 0.84599 |
| sector_K8_center/B | WT2 | 0.61444 → 0.59446 | 0.00000 → 0.00000 | 0.00000 | 0.86155 |
| noise_sigma10/none | P2 | 0.60012 → 0.57493 | 0.03203 → 0.03151 | 0.04880 | 1.00000 |
| noise_sigma10/A | P2 | 0.62984 → 0.59880 | 0.00276 → 0.00276 | 0.00436 | 0.85504 |
| noise_sigma10/none | WT2 | 0.58505 → 0.56671 | 0.02866 → 0.02827 | 0.04880 | 1.00000 |
| noise_sigma10/A | WT2 | 0.61775 → 0.59602 | 0.00251 → 0.00275 | 0.00436 | 0.85504 |

C6δ sensitivity (all 200, equal-weight samples; not used to change the real-C6 abstention threshold):

| δ | Route | Target hit ep72→final | Interferer-only ep72→final | Stream-selection error ep72→final |
|---|---|---:|---:|---:|
| 0 | P2 | 0.635 → 0.620 | 0.000 → 0.000 | 0.000 → 0.005 |
| 10 | P2 | 0.510 → 0.500 | 0.130 → 0.125 | 0.210 → 0.200 |
| -10 | P2 | 0.545 → 0.530 | 0.085 → 0.100 | 0.130 → 0.145 |
| 20 | P2 | 0.385 → 0.380 | 0.245 → 0.235 | 0.395 → 0.390 |
| -20 | P2 | 0.505 → 0.480 | 0.145 → 0.160 | 0.220 → 0.230 |
| 0 | WT2 | 0.590 → 0.570 | 0.005 → 0.005 | 0.000 → 0.005 |
| 10 | WT2 | 0.480 → 0.470 | 0.100 → 0.110 | 0.210 → 0.200 |
| -10 | WT2 | 0.505 → 0.480 | 0.085 → 0.090 | 0.130 → 0.145 |
| 20 | WT2 | 0.380 → 0.375 | 0.215 → 0.220 | 0.395 → 0.390 |
| -20 | WT2 | 0.460 → 0.435 | 0.135 → 0.140 | 0.220 → 0.230 |

- The front-end SI-SDRi of the final model is slightly higher, but the C5 understanding metrics do not exceed ep72; this result is kept as is, and the front-end checkpoint is not reselected.
- The fixed two-stream Audio LoRA runs 22029676/22029677 have been submitted with the original configuration and are running; the final degradation conclusion awaits verification of the outputs. The C3 failure within the overall §6 is unchanged.

## Final Audio LoRA checks completed
- Both streams 22029676/22029677 COMPLETED/0:0, 200 items each; audio SHA256, prompts and labels exactly identical to the same-stream baseline. Frozen scoring 22030055 and paired analysis 22030132 both COMPLETED/0:0; all input and primary-metric consistency assertions pass.
- C5: target-any 0.620→0.835, paired difference +0.215 [0.150,0.275]; interferer-only 0→0.005; pair flip 0.300→0.630.
- Real C6 naturally weighted interferer-only: 8 sectors none 0.05051→0.05882, A 0.00924→0.00924, B 0→0; σ10 none 0.03151→0.03995, A 0.00276→0.00343. All five pass the P2+0.01 point-estimate gate; the C5 flip gate passes.
- All additional interferer hits are in the actually misrouted group; interferer hits in the correctly routed groups are all 0. Full entity counts, the four binding classes, results for both halves and intervals are in `lora_v2_runs.md` §25. The increase in risk cannot be denied, but it does not support attributing it directly to degraded spatial binding under correct routing.
- The remaining SpatialNet gates all pass; overall §6 still fails because of the existing C3 failure; no checkpoint reselection, no threshold re-tuning, Whisper+T2-LoRA not started, final SLURP test not used.
- New LoRA outputs: `outputs/c5_final/lora_step440_20261107/`, `outputs/c6_real_final/lora_step440_20261107/`, `outputs/lora_step1_checks/final_spatialnet_ep100_20261107/`; all original results kept.
