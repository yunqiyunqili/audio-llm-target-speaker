# 流程与模型调用方式（只写实际存在的内容；2026-11-05 更新）

标"**未实现**"的部分目前没有代码，也没有结果。所有 GPU 和重 CPU 任务都通过 sbatch 或 srun 提交（account <SLURM_ACCOUNT>）。

## 0. 总体流程（目标）与现状
```
SLURP 近讲语音 ──(配对规则 + 说话人划分)──> 句子对 ──(SMS-WSJ 6 麦 RIR 空间化，8 kHz)──> 2 人空间混合
   ├─ 条件 C0–C7（docs/conditions.md）
   ├─ 分离前端（BLSTM+MVDR / FaSNet-TAC / ...）──> 分离后的语音 ──> Qwen2.5-Omni ──> 意图 / 实体
   └─ 直接把混合语音送给 Qwen2.5-Omni（不经过分离）──> 意图 / 实体
```

| 环节 | 现状 |
|---|---|
| SLURP 数据（commit 8eb1654 + Zenodo 4274930 slurp_real） | 已下载、已解压、已统计 |
| 配对规则 + 说话人划分 | 正式清单 v2 已生成（`split_v2_seed2026`，只有录音清单和全部候选对，**未抽最终实验对**，未生成音频） |
| 空间化混合（SLURP × SMS-WSJ RIR） | **未实现** |
| 分离前端在 SMS-WSJ 上推理 | BLSTM+MVDR、FaSNet-TAC 已完成评估复现；在 SLURP 混合上推理**未实现** |
| Qwen2.5-Omni 单条推理（转写） | 已跑通（作业 22003339） |
| Qwen2.5-Omni 结构化 SLU（意图 + 实体） | 20 条 devel 试跑完成（作业 22010083），并已离线重评分（docs/c1_semantic_plan.md） |
| 分离前端 → Qwen 的级联 | **未实现** |
| 批量推理、结果汇总、SLU-F1 | **未实现** |

## 1. 数据
- SLURP 标注：`data/slurp/annotations/8eb1654/{train,devel,test}.jsonl`、`metadata.json`。
- SLURP 音频：`data/slurp/audio/slurp_real/*.flac`，近讲录音全部是 16 kHz、单声道、PCM_16。
- 时长表：`results/slurp/headset_durations.csv`，由 `scripts/slurp/slurp_audio_stats.py` 生成（srun）。
- SMS-WSJ：只读使用 `../` 下现有的数据和 RIR，8 kHz、6 麦。几何统计：`scripts/sms_wsj/sms_wsj_geometry_stats.py` → `results/sms_wsj_geometry/`。
- 来源和版本说明：`docs/provenance.md`。

## 2. 配对与划分
- 配对规则：同一 split、同一意图、实体类型多重集合相同且非空、实体值不同、说话人不同。实现见 `scripts/slurp/make_slurp_speaker_heldout.py: pairs_in_split`。
  - v2 起分组用 scenario_action；早期统计脚本用的是 jsonl 的 `intent` 字段，见第 6 节。
- 正式版本：`scripts/slurp/make_slurp_split_v2.py` → 共享盘 `data/slurp/manifests/split_v2_seed2026/`、`configs/slurp_split_v2_seed2026.json`、`results/slurp/split_v2_stats.json`。旧的 `heldout_seed2026/` 已废弃（见 docs/provenance.md）。
- 统计脚本（都用 `envs/qwen_omni/bin/python -I` 在 srun 的 <CPU_PARTITION> 上运行）：
  - `scripts/slurp/make_slurp_speaker_heldout.py`：seed2026、只留出 test 的版本（已生成清单，在共享盘）。
  - `scripts/slurp/compare_slurp_speaker_splits.py`：test 20/30/40% 的比较，结果 `results/slurp/split_comparison.json`。
  - `scripts/slurp/compare_slurp_split_trainsent_devel.py`：devel 只用 train 句子的方案，结果 `results/slurp/split_trainsent_devel_f0.4.json`。
  - `scripts/slurp/compare_slurp_devel_mixed.py`：devel = 官方 devel 句子 + 从 train 留出的 25/30/35/40%，结果 `results/slurp/split_devel_mixed_f0.4.json`。
- 每条录音最多用 3 次时的配对数：用 0/1 整数规划精确求解（`scipy.optimize.milp`）。

## 3. 分离前端
| 模型 | 调用方式 | 环境 | 作业配置 | 实测 |
|---|---|---|---|---|
| BLSTM+MVDR | `scripts/sms_wsj/repro_sms_wsj_test.py --model blstm_mvdr`：import 原 `train.py` 的 `evaluate()`，加载 `../.../best.pt`（只读软链接） | `<CONDA_ROOT>/envs/tfgridnet`（不改动） | `scripts/sms_wsj/repro_blstm_mvdr_test.sbatch`：<GPU_PARTITION> / 1×A100 / 4 CPU / 32 GB / 1 h | 作业 21997819；test_eval92 SI-SDR −1.0428 dB，与记录逐字节一致 |
| FaSNet-TAC | 同上，`--model fasnet_tac` | 同上 | `scripts/sms_wsj/repro_fasnet_tac_test.sbatch`，同上 | 作业 21997820；SI-SDR 9.4404 dB，逐字节一致 |
| BLSTM / oracle / cACGMM 在不同参考信号下的 SI-SDR | `scripts/sms_wsj/rev_ref_sisdr.py` | 同上 | `scripts/sms_wsj/rev_ref_sisdr.sbatch`：<GPU_PARTITION> / 1×A100 / 4 CPU / 32 GB / 1 h | 作业 21997977；见 `docs/log/day1_followup2.md` |
| TF-GridNet、SpatialNet | 只记录了 checkpoint（`docs/checkpoints.md`）；训练作业由原脚本管理，不动 | — | — | 在本项目中的推理**未实现** |
| 分离前端在 SLURP 空间混合上推理 | **未实现** | | | |

## 4. Qwen2.5-Omni-7B
- 环境：`envs/qwen_omni`（→ <SHARED_ROOT>/envs/qwen_omni），Python 3.10.22、torch 2.6.0+cu124、torchaudio 2.6.0+cu124、torchvision 0.21.0+cu124、transformers 4.52.4、accelerate 1.7.0、qwen-omni-utils 0.0.8。完整依赖见 `docs/env_qwen_omni.txt`，建环境脚本 `scripts/qwen/build_env_qwen_omni.sh`。其中 torchvision 是后来补装的，因为 processor 会加载视频处理器。
- 权重：commit ae9e1690543ffd5c0221dc27f79834d0294cba00，本地 snapshot 目录 `checkpoints/hf/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690…`。
- 调用方式：
```python
model_dir = $HF_HOME/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/<commit>    # 必须用本地目录，见下方注意
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='sdpa', local_files_only=True)
model.disable_talker()                                    # 只输出文本
processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True)
conv = [system: Qwen 默认 system prompt, user: [audio, text prompt]]
text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
batch = processor(text=text, audio=[wav_16k_float32_mono], sampling_rate=16000, return_tensors='pt', padding=True)
ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=N, do_sample=False, use_audio_in_video=False)
out = processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
```
- 环境变量：`HF_HOME=<SHARED_ROOT>/checkpoints/hf`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONNOUSERSITE=1`。
- 输入：16 kHz 单声道 float32，由 WhisperFeatureExtractor 转成 128 维 mel 特征。2.432 s 的音频得到的 `input_features` 是 [1,128,30000]（补齐到 30 s）。
- 8 kHz 版本（C1b）：scipy `resample_poly` 先 1/2 再 2/1，截回原长度，再按 16 kHz 输入。
- 注意事项（实测得到）：
  1. 用 repo id 加 revision 加载时，transformers 4.52.4 查找 `spk_dict.pt` 不会使用传入的 revision，离线又没有 `refs/main`，所以会失败（作业 21999394）。改用本地目录后解决。
  2. 没装 torchvision 时，加载 processor 会报 ImportError（作业 21999584）。
- 脚本：
  - 单条转写：`scripts/qwen/qwen_omni_single.py` + `.sbatch`（<GPU_PARTITION> / 1×A100-40GB / 4 CPU / 48 GB / 30 min）。
  - CPU 预检（不加载权重）：`scripts/qwen/qwen_omni_precheck.py`。
  - 结构化 SLU：`scripts/qwen/qwen_omni_c1_semantic.py` + `.sbatch`（作业 22010083）；离线重评分 `scripts/qwen/rescore_c1_semantic.py RUN_DIR`。
- 实测结果（作业 22003339，A100-PCIE-40GB，SLURP test slurp_id 0，2.432 s，参考 "wake me up at five am this week"）：

| | 加载时间 | 加载峰值显存 | 推理时间 | 推理峰值显存 | 输出 |
|---|---|---|---|---|---|
| C1 原始 | 46.8 s（模型 + processor） | 20.83 GiB | 3.27 s（含首次调用开销） | 16.72 GiB | Wake me up at 5 am this week. |
| C1b 8k→16k | — | — | 0.33 s | 16.72 GiB | Wake me up at 5 am this week. |

  - 作业总用时 2 min 07 s，MaxRSS 26.4 GB。只有 1 条样本。

## 5. 评估
- 分离：PIT SI-SDR / SI-SDRi，复用原训练代码中的实现。
- SLU：解析、规范化和打分在 `scripts/qwen/slu_normalize.py`（意图准确率、实体 P/R/F1、规范化后的 WER），规则见 `docs/c1_semantic_plan.md`。SLURP 官方 SLU-F1 **未实现**。

## 6. 已知问题
- SLURP jsonl 的 `intent` 字段有 229 个句子与 `scenario_action` 不一致。C1 语义测试用 scenario_action；v2 划分已改用 scenario_action；v2 之前的统计结果（split_comparison 等）用的是 `intent` 字段，未重算。
