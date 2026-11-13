# 复现 BLSTM+MVDR 和 FaSNet-TAC 的 SMS-WSJ 测试指标（方案，尚未提交）

## 目标
用已有的 `best.pt` 在 `test_eval92`（1,332 条）上重新推理，确认能得到与记录一致的数值：

| 模型 | 记录 SI-SDR | 记录 SI-SDRi | 记录 mixture SI-SDR | 来源 |
|---|---|---|---|---|
| BLSTM+MVDR | −1.0427572 | 4.4121033 | −5.4548605 | `MC/blstm_mvdr/exp/sms_blstm_mvdr_v1/test_summary.json` |
| FaSNet-TAC | 9.4404093 | 14.8952699 | −5.4548605 | `MC/fasnet_tac/exp/sms_fasnet_tac_v1/test_summary.json` |

这是**评估复现**（同一权重、同一代码、同一数据），不重新训练。

## 做法
- `scripts/sms_wsj/repro_sms_wsj_test.py` 直接 import 原始的 `<model>/train.py`，调用它自己的 `evaluate()`，不改写任何评估逻辑。
- 原 `evaluate()` 会把结果写进 run_dir，并对 `run_dir/best.pt` 做 hash。因此脚本把 run_dir 指向 `outputs/repro_sms_wsj/<model>_<jobid>/`，该目录里只放一个指向原 `best.pt` 的软链接。为避免写入 `../`，设置了 `-B` 和 `PYTHONDONTWRITEBYTECODE`。实际检查的路径和时间范围见 `docs/log/day1_report.md` 的“对 `../` 是否有写入的实际检查”。
- 推理前的检查（GPU 作业中也会再做一次）：
  1. `best.pt` 的 sha256 与记录一致；
  2. checkpoint 中的 config 与 `run_config.json` 一致；selected epoch 一致；
  3. 用原 `configuration()` 重算当前代码和 ESPnet/TAC 源码的 hash、audit、manifest hash，与 checkpoint 中存的值一致（确认训练后代码和数据没有改动）。
- 推理后：对比均值，以及逐条对比原 `test_utterances.jsonl` 中的 SI-SDR、SI-SDRi、mixture SI-SDR 和 permutation，结果写入 `comparison.json`。
- 判定：均值差 ≤ 1e-3 dB，逐条最大差 ≤ 1e-2 dB，且 permutation 全部相同 → match，退出码 0；否则退出码 3。实际差值无论是否 match 都会记录。原运行设置了确定性 cuDNN、关闭 TF32，所以预期接近 bit-exact；但如果分到的节点驱动不同，可能出现微小差异，以实测为准。

## 已在登录节点完成的验证（只做 CPU 检查，不推理）
`python -B scripts/sms_wsj/repro_sms_wsj_test.py --model {blstm_mvdr,fasnet_tac} --check-only`：两个模型的 5 项 provenance 检查全部为 true，test 集 1,332 条 → `CHECK_ONLY_PASS`。

## 作业配置（未提交）
| | BLSTM+MVDR | FaSNet-TAC |
|---|---|---|
| 脚本 | `scripts/sms_wsj/repro_blstm_mvdr_test.sbatch` | `scripts/sms_wsj/repro_fasnet_tac_test.sbatch` |
| account / partition | <SLURM_ACCOUNT> / <GPU_PARTITION>（与原训练相同） | 同左 |
| 资源 | 1× A100，4 CPU，32 GB RAM，时限 1 h | 同左 |
| 环境 | `<CONDA_ROOT>/envs/tfgridnet`（与原运行相同，torch 2.5.1+cu121） | 同左 |
| 原 test 推理耗时 | 73 s | 91 s |
| 原峰值显存 | 0.43 GiB | 4.68 GiB |
| 输出 | `outputs/repro_sms_wsj/blstm_mvdr_<jobid>/` | `outputs/repro_sms_wsj/fasnet_tac_<jobid>/` |
| 日志 | `outputs/logs/repro_blstm_mvdr_<jobid>.out` | `outputs/logs/repro_fasnet_tac_<jobid>.out` |
| 新增磁盘 | 约 0.3 MB（逐条 jsonl + summary + 软链接） | 同左 |

提交命令（需要你确认后再执行，在项目根目录运行）：
```
sbatch scripts/sms_wsj/repro_blstm_mvdr_test.sbatch
sbatch scripts/sms_wsj/repro_fasnet_tac_test.sbatch
```
