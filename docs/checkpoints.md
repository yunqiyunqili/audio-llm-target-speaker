# 多通道分离前端 checkpoint 核查

核查时间：2026-11-03 约 12:40 EST（Slurm 状态来自本次 `squeue`/`sacct`）。所有路径只读，未修改。
根目录：`MC=<SEPARATION_REPO>/multichannel`。

## 共同协议（四个模型相同）

- 数据：SMS-WSJ 官方 split，`train_si284` 33,561 / `cv_dev93` 982 / `test_eval92` 1,332 条混合，manifest 在 `MC/data/sms_wsj_direct_v1/`，`audit.json` sha256 `4b716da2…3573c3`（四个 run_config 中的值一致）。
- 采样率 **8 kHz**；2 个说话人；6 通道输入（顺序 0–5）。
- 目标：每个说话人在**参考麦 mic 0 的直达声**（几何法生成，T60=0 的 RIR），不是 `speech_source`、`early` 或混响 image。
- 输入格式：mixture `[B, T, 6]` float32（单样本为 `[T, 6]`）；输入先除以一个全局标量 std（跨时间和通道），保留通道间的幅度关系；输出乘回该 scale。
- 输出格式：`[B, 2, T]` 时域波形，即两个说话人的 mic-0 直达声估计，顺序任意（PIT）。
- 训练：每条每 epoch 随机裁 4 s（seed 2026，由 sha256(seed,epoch,split,id) 决定），batch 2，FP32，Adam 1e-3，100 epoch 上限；dev/test 用完整长度。
- 测试指标：zero-mean PIT SI-SDR、mic-0 mixture SI-SDR、SI-SDRi，在全部 1,332 条上取平均（`tfgridnet/objectives.py: scores`）。mixture SI-SDR 基线 = −5.4549 dB。
- 环境：`<CONDA_ROOT>/envs/tfgridnet`（torch 2.5.1+cu121，CUDA 12.1），GPU 为 A100-PCIE-40GB。
- 每个 checkpoint（`best.pt`/`latest.pt`）的 key：`model, optimizer, scheduler, config, history, epoch, best, complete, python/numpy/torch/cuda_rng`。用 `torch.load(..., weights_only=False)` 加载。

## 总表

| 模型 | 运行目录 | 参数量 | 前端 | 训练状态 | 最佳 epoch | 完整 test（1,332 条） |
|---|---|---|---|---|---|---|
| BLSTM mask + Souden MVDR | `MC/blstm_mvdr/exp/sms_blstm_mvdr_v1` | 6,111,011 | STFT 512/128 Hann | **已完成** 100/100 | 62（min dev mask MSE） | SI-SDR **−1.0428**，SI-SDRi **4.4121** dB |
| FaSNet-TAC | `MC/fasnet_tac/exp/sms_fasnet_tac_v1` | 2,691,279 | 时域 4 ms 窗 / 2 ms hop | **已完成** 100/100 | 98（min dev −SI-SDR） | SI-SDR **9.4404**，SI-SDRi **14.8953** dB |
| TF-GridNet 6ch | `MC/tfgridnet/exp/sms_direct_6ch_v1` | 5,498,440 | STFT 256/64 sqrt-Hann | **训练中** 9/100（job 21989292 RUNNING） | 9（目前） | **无**（尚未产出） |
| SpatialNet-small | `MC/spatialnet/exp/sms_spatialnet_small_v1` | 1,191,092 | STFT 256/128 Hann | **训练中断等待续训** 43/100（续训 job 21995307 PENDING） | 42（目前） | **无**（尚未产出） |

“参数量”取自训练日志中的 `parameters` 字段；BLSTM 和 FaSNet 的数值与我加载 state_dict 后的统计一致。
另有 TF-GridNet **1ch** 对照：`MC/tfgridnet/exp/sms_direct_1ch_v1`，9/100 epoch，job 21989293 RUNNING，参数 5,494,120（非本次重点）。

## 各模型细节

### BLSTM + MVDR
- 代码：`MC/blstm_mvdr/{model.py,train.py,dataset.py}`；ESPnet `DNN_Beamformer`，commit `9c5ec34f`。
- 配置：3 层 BLSTM，每方向 300，proj 320，3 个 sigmoid mask 头（spk1/spk2/noise），Souden MVDR，ref mic 0，diag loading 1e-7，mask floor 1e-6。
- 监督：对**混响** component（early+tail、noise）的 power IRM 做 PIT MSE。MVDR 不参与反传。评估时才用直达声 target，所以训练目标和评估目标不同。
- 日志 `MC/logs/blstm_train_21977529.out`（epoch 1–95），`blstm_train_21988894.out`（epoch 96–100 + test，COMPLETED）。
- `best.pt` sha256 `eebb4385…07882f`，与 `test_summary.json` 记录一致（已核对）。
- 训练曲线：dev SI-SDR 从 −3.10 dB（epoch 1）升到 −1.11 dB（epoch 62）后不再变化；最后 LR 降到 1.5e-8。
- 平均 epoch：训练 1,443 s，验证 74 s；峰值显存 0.43 GiB。

### FaSNet-TAC
- 代码：`MC/fasnet_tac/{model.py,train.py,objectives.py}`；官方 `yluo42/TAC` commit `e3373b73`，`FaSNet_TAC` 未改动。
- 配置：enc_dim 64，feature_dim 64，hidden 128，4 层 DPRNN-TAC，segment 50，win 4 ms，context 16 ms，sr 8000；`num_mic=0` sentinel 表示固定 6 麦。
- 损失：官方 `batch_SDR_torch` 的负 PIT SI-SDR，在有效长度内计算。
- 日志 `MC/logs/fasnet_train_21977642.out`（epoch 1–57），`fasnet_train_21988907.out`（epoch 58–100 + test，COMPLETED，日志最后写入于 2026-11-03 11:49）。
- `best.pt` sha256 `0ce3bef5…9106cf`，与记录一致。最后一轮 dev SI-SDR 9.52 dB，最佳 9.53 dB（epoch 98）。
- 平均 epoch：训练 2,404 s，验证 68 s；峰值显存 4.68 GiB。

### TF-GridNet 6ch
- 代码：`MC/tfgridnet/{model.py,train.py,objectives.py}`；ESPnet `TFGridNetV2`，commit `9c5ec34f`。
- 配置：n_fft 256，stride 64，4 层，LSTM 192，4 heads，emb 48，PReLU；损失为 PIT Wav+Mag+MC（论文 Eq.12）。
- 进度：`metrics.jsonl` 有 9 个 epoch；dev SI-SDR 从 10.18 dB（ep1）到 16.42 dB（ep9）；LR 仍为 1e-3。
- 续训链：`continuation_jobs.txt` = 21986967 → 21989292（RUNNING，已跑 7 h，日志显示第 10 epoch 训练进行到 15,400/16,781 步）。
- 平均 epoch：训练 27,777 s（约 7.7 h），验证 1,239 s；峰值显存 14.6 GiB。
- **没有 `test_summary.json`**，不能报告 test 指标。

### SpatialNet-small
- 代码：`MC/spatialnet/{model.py,train.py,objectives.py}`；官方 NBSS commit `cc42fc8a`，`SpatialNet` 未改动。
- 配置：8 层，hidden 96，FFN 192，squeeze 8，4 heads，`full_share=0`，129 频点；Norm('frequency', online=True)；ExponentialLR 0.99，clip 5。
- 进度：43 个 epoch；dev SI-SDR 10.15 dB（ep1）→ 19.30 dB（ep42，best）→ 19.03 dB（ep43）。
- 续训链：21988892（COMPLETED，以 `CONTINUATION_REQUIRED` 退出）→ 21995307（PENDING，原因 Priority）。
- 平均 epoch：训练 6,433 s，验证 75 s；峰值显存 31.95 GiB。
- **没有 `test_summary.json`**。
- 说明：`best.pt` 的 state_dict 元素总数是 2,130,468，不是 1,191,092，原因是 `full_share=0` 让 layer 1–7 共享 layer 0 的 full-band 权重（7 × 133,128 个重复项）。checkpoint 中的 config 和 `run_config.json` 的差异仅为 tuple/list 序列化，内容相同。

## 需要注意的地方

1. BLSTM+MVDR 的 test SI-SDR 是 −1.04 dB。SI-SDRi 为正（4.41 dB），但绝对值远低于另外三个模型，dev 曲线在 epoch 62 后不再变化。这与记录的配置一致（训练用混响 IRM，MVDR 输出应保留参考麦混响，评估对比直达声），但我没有做额外实验来验证这个解释。
2. TF-GridNet 每 epoch 约 8 h，按当前速度跑满 100 epoch 还需要约 30 天墙钟时间。SpatialNet 每 epoch 约 1.8 h，剩余 57 epoch 需要约 4.3 天（不含排队）。
3. 四个模型都是 8 kHz，见 docs/log/day1_report.md 中关于 SLURP 16 kHz 的决策项。
