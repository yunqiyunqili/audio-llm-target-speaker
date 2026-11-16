English translation of frontend_compare_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：frontend_compare_v1.md

# Stronger front-end comparison v1 (frontend-compare-v1; **draft**, 2026-11-05; no jobs submitted)

## Checkpoint inventory (2026-11-05 about 21:10, read-only; for run directories being written, only file metadata was looked at, not content)
Root directory `MC = <SEPARATION_REPO>/multichannel`.

| Run | Status | Basis | Usable? |
|---|---|---|---|
| TF-GridNet 6ch `MC/tfgridnet/exp/sms_direct_6ch_v1` | **training**: job 22009884 (sms-grid-6ch) RUNNING, running for about 21 hours | best.pt / latest.pt / metrics.jsonl last modified 2026-11-05 15:45:53; **no test_summary.json** | no: not completed, no definite test metrics |
| TF-GridNet 1ch `MC/tfgridnet/exp/sms_direct_1ch_v1` | training: job 22010327 RUNNING | latest.pt last modified 18:44:41; no test_summary.json | no; and single-channel |
| SpatialNet-small `MC/spatialnet/exp/sms_spatialnet_small_v1` | **training**: job 22010857 RUNNING, running for about 13 hours | best.pt last modified 20:34:09; no test_summary.json | no |
| TF-GridNet pilot_6ch / smoke_6ch, SpatialNet smoke | completed, but all smoke runs: 1 epoch, only 5 utterances tested | test_summary: SI-SDRi −12.99 / −12.81 / −6.91 dB; checkpoint sha256 starting with `ac9fdef2…`, `b2d36e69…`, `d3ead726…` respectively (from test_summary) | no: pipeline checks only |
| `repos/TFGridNet_8k_random` | training: job 21972919 (tfgridnet-8k-random) RUNNING | single-channel, different data (not SMS-WSJ 6-mic) | no |

- Current epoch counts and validation metrics: this information is in the `metrics.jsonl` files being written, and as required **was not read**. The record of 2026-11-03 (`docs/checkpoints.md`): TF-GridNet 6ch was then at 9/100 epochs, dev SI-SDR 16.42 dB; SpatialNet at 43/100 epochs, best dev 19.30 dB (epoch 42). These two numbers are now out of date.
- Input: both models are 6-channel, 8 kHz, with the mic 0 direct sound as target and output (B, 2, T), the same protocol as FaSNet-TAC.
- **Conclusion: there is currently no TF-GridNet or SpatialNet checkpoint that is "completed with definite metrics"**; this draft cannot be run for now.

## Preconditions for running (all must be met)
1. The run directory has `test_summary.json`, the checkpoint has `complete == True`, and the training job that produced it has ended and is neither running nor queued.
2. The sha256 of `best.pt` matches the record in `test_summary.json`, and the package config matches `run_config.json`.
3. Only completed files are read; running training jobs are not touched.

## Method (the same as the FaSNet-TAC C5 chain, zero-shot)
- Input: the 200 `obs_6ch_8k.wav` of mix_v1. Inference as in `frontend_mix_v1.py`: std normalisation → model → × scale. Output written to `data/mix_v1/separated_v1/<frontend>/`, without overwriting existing files.
- Stream selection: the same as v3. Polarity check with zero-lag correlation → `doa` → take the stream closest to the query azimuth θ1, stream 0 on a tie.
  - **Depends on the oracle azimuth**, as C5 does.
  - Selection uses only the outputs, the observation and the array geometry, not reference signals.
- Separation metrics: two-talker mean SI-SDRi and target-stream SI-SDRi under the direct reference; the references are `outputs/frontend_mix_v1_eval_v2/direct_refs`.
- Stream-selection accuracy: against direct-reference matching, reported on two bases, all samples and excluding [0,15).
- Downstream, evaluated separately on the selected target stream:
  - **P2**: 8→16 kHz → −30 dBFS → frozen P2, the same chain as C5;
  - **Whisper + T2**: the same audio → whisper-large-v3 (`06f233fe…`) → frozen T2, the same chain as C7W-C5.
- Metrics: `eval_c4_v1.score` and `summ`, including the four class proportions, target-any, all target values hit, pair-flip accuracy, M, U, and Whisper WER.

## Verdict (empirical, not significance tests; no test data used)
- Each of the two chains is compared with FaSNet-TAC (a = target-any over the 200 samples):
  - P2: a(new front end + P2) ≥ a(FaSNet-TAC C5) + 0.10, i.e. ≥ 0.395 + 0.10 = **0.495**;
  - Whisper + T2: a(new front end + Whisper + T2) ≥ a(FaSNet-TAC C7W-C5) + 0.10, i.e. ≥ 0.340 + 0.10 = **0.440**.
- Whichever condition is met, that chain is judged "**a stronger front end clearly helps**"; otherwise "it cannot be established that it clearly helps". The two chains are reported separately.
- SMS-WSJ test metrics are used only to confirm whether a checkpoint qualifies (from existing test_summary.json); no new test results are computed, and SLURP test is not used.

## Expected resources (estimates; measurements take precedence)
- Front-end inference (tfgridnet environment): one job per front end, <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 32 GB, 1 hour. 200 short samples (about 1.5–4.6 s) estimated at a few minutes; output about 33 MB.
- Downstream (qwen_omni environment): one job per front end, <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 48 GB, 30 minutes.
  - Estimate: P2 on 200 items about 4–7 minutes (audio input measured at 1.1–1.9 s per item), Whisper about 1 minute, T2 about 3.5 minutes, loading about 1–1.5 minutes, about 10–14 minutes in total.
  - The timeout of 22016221 shows that this estimate may be low. If necessary, P2 and Whisper + T2 can be split into two jobs.
- Nothing needs to be downloaded: Whisper is already available, and so is Qwen.

## Training progress (2026-11-05 21:12; only continuation_jobs.txt and metrics.jsonl were read; checkpoint contents not read, no sha256 computed)
| | TF-GridNet 6ch (`sms_direct_6ch_v1`) | SpatialNet-small (`sms_spatialnet_small_v1`) |
|---|---|---|
| Continuation job chain (continuation_jobs.txt) | 21986967 → 21989292 → 22009884 (RUNNING) | 21988892 → 21995307 → 22010857 (RUNNING) |
| Epochs completed so far | **16** / 100 | **72** / 100 |
| Best validation dev SI-SDR | **17.565 dB, epoch 15** (epoch 16 is 17.537 dB; the minimum dev loss is at epoch 16) | **19.943 dB, epoch 72** (the minimum dev loss is also at epoch 72) |
| dev SI-SDR of the last 5 epochs | 12:17.11, 13:17.26, 14:17.19, 15:17.57, 16:17.54 | 68:19.77, 69:19.65, 70:19.82, 71:19.94, 72:19.94 |
| Mean training time of the last 5 epochs | 27,423 s (about 7.6 hours) | 6,360 s (about 1.8 hours) |
| Last update of metrics.jsonl | 2026-11-05 15:45:53 (3,921 B) | 2026-11-05 20:34:09 (18,784 B) |
| best.pt | 66,211,644 B, 2026-11-05 15:45:53 | 14,618,898 B, 2026-11-05 20:34:09 |
| latest.pt | 66,236,552 B, 2026-11-05 15:45:53 | 14,670,992 B, 2026-11-05 20:34:08 |
| Learning rate of the next epoch | 0.001 | 4.85e-4 |

- The TF-GridNet best.pt and the epoch-16 metrics were written at the same moment, so best.pt very likely corresponds to epoch 16, i.e. selected by dev loss rather than the epoch 15 with the best dev SI-SDR. This is only inferred from modification times and was not confirmed by reading the checkpoint contents.
- Neither model **has completed training or has test metrics**.

## Diagnostic snapshot rules (draft, 2026-11-05; separate approval needed before running)
Used to run a diagnosis with checkpoints under training before training completes. **Snapshot results count only as diagnostics, not as completed models, and no test data are used.**
1. **Selection**: choose the checkpoint with the best validation metric (dev SI-SDR) that has **finished being written**. If the epoch of best.pt is not the one with the best dev SI-SDR, record the difference truthfully; which file is finally used is determined by "the highest dev SI-SDR among files that have finished being written", stated before the run.
2. **Judging "finished being written"**: record the file size and modification time before copying, record them again at least 60 seconds later; only if the two agree and metrics.jsonl has not been updated in that time does it count as finished.
3. **Copying**: read-only copy into a new directory on shared storage, `<SHARED_ROOT>/checkpoints/snapshots/<model>_ep<epoch>_<时间>/`, **writing nothing into the training directory**. After copying, record the size and modification time of the source file once more; they must agree with those before copying, otherwise void and redo; the size of the target file must also equal that of the source.
4. **Record**: original path, epoch, dev SI-SDR, file size, modification time, and the sha256 of the copy (computed on the copy only). Written into a `snapshot.json` and this file.
5. **Load check**: load from the copy; the package config must agree with `run_config.json`; record the value of the `complete` field (expected False).
6. **First check on cv_dev93**: with the same 50 groups as frontend_sanity (`random.Random(2026).sample`), infer in memory, compute SI-SDRi with the official direct references, and check the polarity of the two output streams, with the same rules as v3. This step only confirms that the pipeline is usable; no pass threshold.
7. Then run per the "Method" earlier in this file, with **verdict thresholds unchanged**: P2 ≥ 0.495; Whisper + T2 ≥ 0.440. Results labelled "diagnostic snapshot".

## Diagnostic snapshot: approved execution (2026-11-05; written before copying)
- **Both TF-GridNet 6ch and SpatialNet-small use the `best.pt` at the time of copying.** The actual epoch and validation metrics are read from **the copy** and recorded (`epoch`, `best`, `history` in the package); checkpoint contents are not read from the training directory.
- Snapshot directory: `<SHARED_ROOT>/checkpoints/snapshots/<model>_ep<epoch>_<YYYYmmdd-HHMMSS>/`, containing `best.pt` and `snapshot.json`.
- **These are snapshots during training, for diagnosis only, not representing completed models**; test is not used.
- **Stopping conditions on cv_dev93** (judged per model): if inference errors, the output shape is not (2, T), or the mean direct SI-SDRi over the 50 groups is ≤ 0 (i.e. no better than the mixture), stop and do not proceed to mix_v1.
- After passing:
  1. zero-shot inference on the 200 mix_v1 samples, output to `data/mix_v1/separated_v1_snapshot/<model>_ep<epoch>/`;
  2. evaluate direct SI-SDRi, polarity and stream-selection accuracy per the v3 flow;
  3. select the target stream with the same stream selection as C5, and evaluate with P2 and Whisper + T2 separately.
- Verdict thresholds unchanged: P2 ≥ 0.495; Whisper + T2 ≥ 0.440. The two models are reported separately.
- The working tree must be clean before each job submission; record the commit, node, GPU model and per-item time.

## Diagnostic snapshot results (2026-11-05; **snapshot diagnostics, not completed models**; no test used)
### Snapshots (srun 22016808; rules as above; the size and modification time of the source files did not change across two checks 65 seconds apart and after copying; nothing was written into the training directories)
| Model | Original path | Copy path | epoch | Validation dev SI-SDR | best (selection criterion) | complete | Copy sha256 |
|---|---|---|---|---|---|---|---|
| TF-GridNet 6ch | `MC/tfgridnet/exp/sms_direct_6ch_v1/best.pt` (66,211,644 B, 2026-11-05 15:45:53) | `checkpoints/snapshots/tfgridnet_ep16_20261105-211918/best.pt` | **16** | 17.537 dB | 0.70188 (dev loss, lower is better) | False | `660fe57fbc31e0ed8cae0cd29e9960bc756883f37d45ec05ab626baa54451cf8` |
| SpatialNet-small | `MC/spatialnet/exp/sms_spatialnet_small_v1/best.pt` (14,618,898 B, 2026-11-05 20:34:09) | `checkpoints/snapshots/spatialnet_ep72_20261105-211918/best.pt` | **72** | 19.943 dB | −19.9425 (dev loss) | False | `582b2a3a10275866a4d379c8c43501e6482e41c39e6d6f1131f40f8444309a3c` |

- The TF-GridNet best.pt was selected by dev loss and corresponds to epoch 16, not to epoch 15 with the best dev SI-SDR (17.565 dB).
- The SpatialNet config differs from run_config.json only in the tuple vs list serialisation of 4 model fields, and is exactly equal after a JSON round trip (srun 22016812).

### Front end (job 22016833: <NODE>, A100-PCIE-40GB, 21:37:55–21:43:40, exit code 0; commit `bfe2d24`, DIRTY 0; CPU dry-run srun 22016815)
| | TF-GridNet snapshot | SpatialNet snapshot | FaSNet-TAC (control, completed model) |
|---|---|---|---|
| cv_dev93 50 groups: direct SI-SDRi (mixture SI-SDR −5.61 dB) | **23.16 dB** → pass | **25.57 dB** → pass | 15.05 dB (frontend_sanity_v1) |
| cv_dev93: proportion with negative correlation to the direct reference (stream 0 / stream 1) | 0 / 0 | **1.00 / 1.00** | 1.00 / 0 |
| cv_dev93: stream-selection accuracy, before → after polarity correction | 1.00 → 1.00 | 0.31 → **1.00** | 0.64 → 1.00 |
| cv_dev93: inference time per item | 1.29 s | 0.11 s | — |
| mix_v1 200 samples: direct SI-SDRi (two-talker mean / target stream) | **17.66 / 17.78 dB** | **18.56 / 18.59 dB** | 11.38 / 11.42 dB |
| mix_v1: stream-selection accuracy (all / excluding [0,15)) | 0.99 / **1.00** | 1.00 / **1.00** | 0.99 / 1.00 |
| mix_v1: proportion inverted (stream 0 / stream 1) | 0 / 0 | 1.00 / 1.00 | 1.00 / 0 |

- mix_v1 outputs: `data/mix_v1/separated_v1_snapshot/{tfgridnet_ep16_…,spatialnet_ep72_…}/`, 33 MB each. Evaluation: srun 22016898, results in `outputs/frontend_snapshot_eval/`.
- direct SI-SDRi by angle stratum ([0,15), [15,30), [30,60), [60,180]): TF-GridNet 16.66 / 17.73 / 18.13 / 18.11 dB; SpatialNet 17.45 / 18.95 / 18.85 / 18.97 dB.

### Downstream (stream selection the same as C5, depending on the oracle azimuth; job 22016929 TF-GridNet, <NODE>; 22016930 SpatialNet, <NODE>; both A100-PCIE-40GB; commit `4eb0bba`, no uncommitted dependency files; CPU dry-run srun 22016911)
| Condition | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | M | U | Parse failures / refusals |
|---|---|---|---|---|---|---|---|---|---|
| TF-GridNet snapshot + P2 | 0.510 | 0.010 | 0.480 | **0.595** | 0.510 | 0.29 | 166 | 30 | 1 / 0 |
| SpatialNet snapshot + P2 | 0.530 | 0 | 0.470 | **0.635** | 0.530 | 0.33 | 170 | 29 | 0 / 0 |
| FaSNet-TAC C5 (P2, control) | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 | 82 | 0 / 0 |
| TF-GridNet snapshot + Whisper + T2 | 0.480 | 0.005 | 0.515 | **0.545** | 0.480 | 0.31 | 160 | 36 | 0 / 0 |
| SpatialNet snapshot + Whisper + T2 | 0.510 | 0.005 | 0.485 | **0.590** | 0.510 | 0.32 | 171 | 27 | 1 / 1 |
| FaSNet-TAC C7W-C5 (control) | 0.290 | 0.005 | 0.705 | 0.340 | 0.290 | 0.13 | 109 | 87 | 3 / 3 |
| C4-direct (P2, upper-bound reference) | 0.565 | 0 | 0.435 | 0.670 | 0.565 | 0.36 | 185 | 15 | 0 / 0 |

- Whisper WER against the target sentence: TF-GridNet 0.198 (159 items with WER < 0.5), SpatialNet 0.157 (173 items). Controls: FaSNet-TAC C5 0.452 (109 items), direct reference 0.078.
- Inference time per item (mean / max, in s): TF-GridNet job P2 1.168 / 7.19, Whisper 0.198 / 0.93, T2 1.082 / 1.78; SpatialNet job P2 1.154 / 4.74, Whisper 0.204 / 0.69, T2 1.097 / 1.77.
- Evaluation: `scripts/qwen/eval_snapshot_downstream.py`, results in `outputs/frontend_snapshot_downstream_eval/eval.json`.

### Verdict (thresholds unchanged; **snapshot diagnostics, not completed models**)
| Model | P2: a (threshold 0.495; margin over FaSNet) | Verdict | Whisper + T2: a (threshold 0.440; margin over FaSNet) | Verdict |
|---|---|---|---|---|
| TF-GridNet snapshot (epoch 16) | 0.595 (+0.200) | **a stronger front end clearly helps** | 0.545 (+0.205) | **a stronger front end clearly helps** |
| SpatialNet snapshot (epoch 72) | 0.635 (+0.240) | **a stronger front end clearly helps** | 0.590 (+0.250) | **a stronger front end clearly helps** |

- These are all only diagnostics: the models are still training, the samples are only 100 devel groups, run once, and stream selection depends on the oracle azimuth.
- SpatialNet snapshot + P2 is 0.635, only 0.035 below the 0.670 of C4-direct.
