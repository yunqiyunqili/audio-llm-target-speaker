# 数据与环境来源记录（长期有效）

## SLURP 版本
- 后续所有 SLURP 处理统一使用 GitHub pswietojanski/slurp 的 commit **8eb16545762be97ace75334109d73824217311f1**（简写 8eb1654）的标注，路径 `data/slurp/annotations/8eb1654/`；音频为 Zenodo 4274930 的 slurp_real（md5 已核对）。
- 划分偏离官方的决定和原因见 `docs/log/day1_followup4.md` §6。
- 论文与发布数据的差异**尚未查清**（2026-11-03 核对，见 `docs/log/day1_followup3.md` §7）：
  - 近讲录音：发布数据按唯一文件 34,663 条，论文 Table 2 为 34,603，差 60 条。发布的标注与音频彼此一致（缺失 0、多余 0）。
  - 说话人：metadata.json 中 174 个 usrid，论文 Table 2 为 177，差 3 人。
  - 另有 1 个文件 `audio-1502309360-headset.flac` 同时挂在 slurp_id 15959 和 16142 下（标签不同），按 split 相加会多算 1 条（34,664）；所有派生清单都排除了这个文件。
- 文中报告 SLURP 数量时，以 commit 8eb1654 的发布数据为准，并注明与论文数字不一致。

## conda 在家目录的写入（建 envs/qwen_omni 时产生，2026-11-03 17:18）
`conda create -p <SHARED_ROOT>/envs/qwen_omni` 在家目录写了 3 个文件，均为 conda 自动行为，未手动修改：

| 文件 | 大小 | 内容 |
|---|---|---|
| `~/.conda/environments.txt` | 306 B | 追加一行，登记新环境路径 |
| `~/.conda/aau_token` | 22 B | conda 匿名使用标识，被更新 |
| `~/.conda/aau_token_host` | 8 B | 同上 |

除此之外没有改动现有环境（`<CONDA_ROOT>/envs/*`、`speech_separation/envs/*` 无 17:00 之后修改的文件）。

## Qwen2.5-Omni-7B 权重
- `Qwen/Qwen2.5-Omni-7B` commit ae9e1690543ffd5c0221dc27f79834d0294cba00，本地 snapshot 目录
  `<SHARED_ROOT>/checkpoints/hf/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00`。
- 缓存中没有 `refs/main`；推理脚本直接从这个目录加载（离线），commit 由目录固定。

## 不是本项目提交的作业：22009884（sms-grid-6ch，TF-GridNet 6ch）
- 结论：这个作业不是本项目提交的，看起来是原训练脚本在上一段作业结束时自动续交的。本项目没有动它，也没有动 TF-GridNet / SpatialNet 的任何作业。
- 依据（2026-11-04 只读查询）：
  1. 本项目的提交记录 `outputs/logs/submitted_jobs.txt` 中没有 22009884。
  2. `sacct`：上一段作业 21989292 在 2026-11-04T21:26:43 结束，状态 COMPLETED；22009884 的提交时间正好也是 2026-11-04T21:26:43。两个作业的 WorkDir 都是 `../speech_separation/multichannel`，SubmitLine 相同，都是 `sbatch --parsable --job-name=sms-grid-6ch train.sbatch 6 tfgridnet/exp/sms_direct_6ch_v1`。
  3. `../speech_separation/multichannel/train.sbatch` 第 20–23 行：`train.py` 退出码为 75 时，脚本会执行 `sbatch --parsable --job-name="sms-grid-${channels}ch" train.sbatch ...`，把新作业号追加到 `$run_dir/continuation_jobs.txt`，并打印 `CONTINUATION_SUBMITTED`。
  4. `../speech_separation/multichannel/logs/train_21989292.out` 末尾依次是 epoch 14（dev SI-SDR 17.19 dB）、`CONTINUATION_REQUIRED`、`CONTINUATION_SUBMITTED 22009884`。

## SLURP 划分版本
- **后续统一使用 `data/slurp/manifests/split_v2_seed2026/`**，配置见 `configs/slurp_split_v2_seed2026.json`，说明见 `docs/log/day1_followup4.md` §9。规则：意图为 scenario_action，口径 A，seed 2026，test 40%；devel 为剩余已知性别说话人的 20%，句子为全部官方 devel 句子加 35% 的官方 train 句子。
- `data/slurp/manifests/heldout_seed2026/`（A_*、S_*、S_dedup_*）**已废弃**（2026-11-04），目录中已加 `DEPRECATED.md`，文件没有删除。`configs/slurp_heldout_speakers_seed2026.json` 同样只供追溯。
- 敏感性清单（只要求 status=correct）以后要用时，按 v2 的规则重新生成。
- 已接受的局限（2026-11-04 用户决定）：v2 的 devel 只有 11 个有录音的说话人、428 条录音；按每条录音最多用 3 次、同性别、时长差 ≤1 s 计，有 197 对。devel 与 train 的实体类型总变差距离为 0.264，大于 test 与 train 之间的 0.201。
- 语义相近的实体类型（例如 time / timeofday）是否混用，暂不检查；如果 C1 判错样本里出现这类问题，再报告。
