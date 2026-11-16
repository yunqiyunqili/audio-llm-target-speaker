English translation of checkpoints.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：checkpoints.md

# Checkpoint audit of the multichannel separation front ends

Audit time: 2026-11-03 about 12:40 EST (Slurm status from `squeue`/`sacct` at that time). All paths read-only, nothing modified.
Root directory: `MC=<SEPARATION_REPO>/multichannel`.

## Common protocol (the same for all four models)

- Data: official SMS-WSJ splits, `train_si284` 33,561 / `cv_dev93` 982 / `test_eval92` 1,332 mixtures, manifests in `MC/data/sms_wsj_direct_v1/`, `audit.json` sha256 `4b716da2…3573c3` (the value agrees in all four run_configs).
- Sample rate **8 kHz**; 2 talkers; 6-channel input (order 0–5).
- Target: each talker's **direct sound at the reference mic 0** (generated geometrically, RIR with T60=0), not `speech_source`, `early` or the reverberant image.
- Input format: mixture `[B, T, 6]` float32 (`[T, 6]` for a single sample); the input is first divided by a global scalar std (across time and channels), preserving the inter-channel amplitude relations; the output is multiplied back by that scale.
- Output format: `[B, 2, T]` time-domain waveforms, i.e. estimates of the two talkers' mic-0 direct sound, in arbitrary order (PIT).
- Training: a random 4 s crop per utterance per epoch (seed 2026, determined by sha256(seed,epoch,split,id)), batch 2, FP32, Adam 1e-3, at most 100 epochs; dev/test use full length.
- Test metrics: zero-mean PIT SI-SDR, mic-0 mixture SI-SDR, SI-SDRi, averaged over all 1,332 utterances (`tfgridnet/objectives.py: scores`). Mixture SI-SDR baseline = −5.4549 dB.
- Environment: `<CONDA_ROOT>/envs/tfgridnet` (torch 2.5.1+cu121, CUDA 12.1), GPU A100-PCIE-40GB.
- Keys of each checkpoint (`best.pt`/`latest.pt`): `model, optimizer, scheduler, config, history, epoch, best, complete, python/numpy/torch/cuda_rng`. Loaded with `torch.load(..., weights_only=False)`.

## Summary table

| Model | Run directory | Parameters | Front end | Training status | Best epoch | Full test (1,332 utterances) |
|---|---|---|---|---|---|---|
| BLSTM mask + Souden MVDR | `MC/blstm_mvdr/exp/sms_blstm_mvdr_v1` | 6,111,011 | STFT 512/128 Hann | **completed** 100/100 | 62 (min dev mask MSE) | SI-SDR **−1.0428**, SI-SDRi **4.4121** dB |
| FaSNet-TAC | `MC/fasnet_tac/exp/sms_fasnet_tac_v1` | 2,691,279 | time domain, 4 ms window / 2 ms hop | **completed** 100/100 | 98 (min dev −SI-SDR) | SI-SDR **9.4404**, SI-SDRi **14.8953** dB |
| TF-GridNet 6ch | `MC/tfgridnet/exp/sms_direct_6ch_v1` | 5,498,440 | STFT 256/64 sqrt-Hann | **training** 9/100 (job 21989292 RUNNING) | 9 (so far) | **none** (not yet produced) |
| SpatialNet-small | `MC/spatialnet/exp/sms_spatialnet_small_v1` | 1,191,092 | STFT 256/128 Hann | **training interrupted, awaiting continuation** 43/100 (continuation job 21995307 PENDING) | 42 (so far) | **none** (not yet produced) |

"Parameters" is taken from the `parameters` field in the training logs; for BLSTM and FaSNet the values agree with my own count after loading the state_dict.
There is also a TF-GridNet **1ch** control: `MC/tfgridnet/exp/sms_direct_1ch_v1`, 9/100 epochs, job 21989293 RUNNING, 5,494,120 parameters (not the focus here).

## Details per model

### BLSTM + MVDR
- Code: `MC/blstm_mvdr/{model.py,train.py,dataset.py}`; ESPnet `DNN_Beamformer`, commit `9c5ec34f`.
- Configuration: 3-layer BLSTM, 300 per direction, proj 320, 3 sigmoid mask heads (spk1/spk2/noise), Souden MVDR, ref mic 0, diag loading 1e-7, mask floor 1e-6.
- Supervision: PIT MSE on the power IRM of the **reverberant** components (early+tail, noise). MVDR takes no part in backpropagation. The direct-sound target is used only in evaluation, so the training and evaluation targets differ.
- Logs `MC/logs/blstm_train_21977529.out` (epochs 1–95), `blstm_train_21988894.out` (epochs 96–100 + test, COMPLETED).
- `best.pt` sha256 `eebb4385…07882f`, matching the record in `test_summary.json` (checked).
- Training curve: dev SI-SDR rises from −3.10 dB (epoch 1) to −1.11 dB (epoch 62) and then stops changing; the final LR is down to 1.5e-8.
- Mean per epoch: training 1,443 s, validation 74 s; peak GPU memory 0.43 GiB.

### FaSNet-TAC
- Code: `MC/fasnet_tac/{model.py,train.py,objectives.py}`; official `yluo42/TAC` commit `e3373b73`, `FaSNet_TAC` unchanged.
- Configuration: enc_dim 64, feature_dim 64, hidden 128, 4 DPRNN-TAC layers, segment 50, win 4 ms, context 16 ms, sr 8000; the `num_mic=0` sentinel denotes a fixed 6 mics.
- Loss: negative PIT SI-SDR of the official `batch_SDR_torch`, computed within the valid length.
- Logs `MC/logs/fasnet_train_21977642.out` (epochs 1–57), `fasnet_train_21988907.out` (epochs 58–100 + test, COMPLETED, log last written 2026-11-03 11:49).
- `best.pt` sha256 `0ce3bef5…9106cf`, matching the record. Last-epoch dev SI-SDR 9.52 dB, best 9.53 dB (epoch 98).
- Mean per epoch: training 2,404 s, validation 68 s; peak GPU memory 4.68 GiB.

### TF-GridNet 6ch
- Code: `MC/tfgridnet/{model.py,train.py,objectives.py}`; ESPnet `TFGridNetV2`, commit `9c5ec34f`.
- Configuration: n_fft 256, stride 64, 4 layers, LSTM 192, 4 heads, emb 48, PReLU; loss PIT Wav+Mag+MC (Eq.12 of the paper).
- Progress: `metrics.jsonl` has 9 epochs; dev SI-SDR from 10.18 dB (ep1) to 16.42 dB (ep9); LR still 1e-3.
- Continuation chain: `continuation_jobs.txt` = 21986967 → 21989292 (RUNNING, 7 h so far; the log shows epoch 10 training at step 15,400/16,781).
- Mean per epoch: training 27,777 s (about 7.7 h), validation 1,239 s; peak GPU memory 14.6 GiB.
- **No `test_summary.json`**; test metrics cannot be reported.

### SpatialNet-small
- Code: `MC/spatialnet/{model.py,train.py,objectives.py}`; official NBSS commit `cc42fc8a`, `SpatialNet` unchanged.
- Configuration: 8 layers, hidden 96, FFN 192, squeeze 8, 4 heads, `full_share=0`, 129 frequency bins; Norm('frequency', online=True); ExponentialLR 0.99, clip 5.
- Progress: 43 epochs; dev SI-SDR 10.15 dB (ep1) → 19.30 dB (ep42, best) → 19.03 dB (ep43).
- Continuation chain: 21988892 (COMPLETED, exited with `CONTINUATION_REQUIRED`) → 21995307 (PENDING, reason Priority).
- Mean per epoch: training 6,433 s, validation 75 s; peak GPU memory 31.95 GiB.
- **No `test_summary.json`**.
- Note: the total number of elements in the state_dict of `best.pt` is 2,130,468, not 1,191,092, because `full_share=0` makes layers 1–7 share the full-band weights of layer 0 (7 × 133,128 duplicate entries). The only difference between the config in the checkpoint and `run_config.json` is tuple/list serialisation; the content is the same.

## Points to note

1. The test SI-SDR of BLSTM+MVDR is −1.04 dB. The SI-SDRi is positive (4.41 dB), but the absolute value is far below the other three models, and the dev curve stops changing after epoch 62. This is consistent with the recorded configuration (training with the reverberant IRM, the MVDR output is expected to keep the reference-mic reverberation, and evaluation compares with the direct sound), but I did not run any additional experiment to verify this explanation.
2. TF-GridNet takes about 8 h per epoch; at the current speed, finishing 100 epochs would take about another 30 days of wall-clock time. SpatialNet takes about 1.8 h per epoch; the remaining 57 epochs need about 4.3 days (excluding queueing).
3. All four models are 8 kHz; see the decision item about SLURP at 16 kHz in docs/log/day1_report.md.
