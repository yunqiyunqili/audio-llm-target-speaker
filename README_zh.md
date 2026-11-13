[English](README.md) | **中文**

# 空间音频口语理解：音频 LLM 与级联 ASR 的对比（SLURP + FSC）

## 问题
两人同时说话、只知道目标说话人的大致方向时，怎样把目标的话交给大模型做口语理解（意图 + 实体）？本项目用多通道空间分离（SpatialNet）+ 方向选路得到目标语音，再比较两条路线：
- **音频路线**：分离出的目标语音直接输入 Qwen2.5-Omni-7B，并用音频 LoRA 微调；
- **级联路线**：Whisper large-v3 转写后，由文本 LoRA（T2-LoRA）微调的 Qwen 做 SLU。

预先写定的主要比较为 **音频 LoRA − Whisper + T2-LoRA**（target-any，配对），在单人干净（C1）、单人混响（C1-R）、两人分离选路后（C5）三个条件上检验，Bonferroni 校正（98.33% 区间下界 > 0 才算显著）。

## 结论
- **SLURP test**（只运行一次，按冻结协议）：三项主要比较都显著，音频 LoRA 优于 Whisper + T2-LoRA，但幅度不大（+0.03 至 +0.07）。C1 的下界接近 0；C5 的自然夹角加权版本跨 0（只作描述）。
- **FSC 跨数据集检验**：**主要比较未重现**，三项都不显著，点估计在 −0.014 至 +0.017 之间。可能原因（标签映射低估、FSC 封闭且简单、前端更差）**均未验证**。
- 两种 LoRA 相对各自基线在 SLURP 上都提升约 +0.25 至 +0.28，在 FSC 上只提升 +0.06 至 +0.10。
- **干扰者风险**：在真实系统路径上（模拟的粗方向 / 方向噪声），SLURP 上两种 LoRA 的"只命中干扰者"都比各自基线高约 +0.02；拒答后降到约 +0.01。FSC 上路由正确时的干扰者命中有 92–98% 来自意图映射补出的默认值，因此 FSC 上两种 LoRA 的干扰者风险差异主要反映映射方式，不能解释为模型行为差异。
- 所有空间条件都是模拟混合。**这些结果不推断到其他数据集或真实录音。**

## 系统流程
```
6 通道混合（8 kHz）
 → SpatialNet-small ep100 分离（两路）
 → 极性修正（零时延 mic0 相关）
 → PHAT 方向估计（每路一个 DOA）
 → 选路：离询问方向最近的一路（v3）
 → 拒答（可选）：A = 两路方向 margin < 17°；B = 两路都落在询问扇区内
 → 选中路：重采样到 16 kHz，电平 −30 dBFS
 → 音频路线：Qwen2.5-Omni + P2（原始 / 音频 LoRA）
   级联路线：Whisper large-v3 → Qwen + T2（原始 / T2-LoRA）
 → JSON（transcript、intent、entities）→ slu-score-v2 / binding-score-v1
```

## 主要结果（音频 LoRA − Whisper + T2-LoRA，target-any）

| 数据集 / 条件 | n | 音频 LoRA | Whisper+T2-LoRA | 差值 | 98.33% 区间 | 显著 |
|---|---|---|---|---|---|---|
| SLURP test C1 | 1,183 | 0.848 | 0.815 | +0.033 | [+0.006, +0.060] | 是 |
| SLURP test C1-R | 1,183 | 0.806 | 0.751 | +0.055 | [+0.025, +0.086] | 是 |
| SLURP test C5 | 400 | 0.755 | 0.683 | +0.073 | [+0.025, +0.120] | 是 |
| FSC clean | 1,240 | 0.909 | 0.923 | −0.014 | [−0.049, +0.020] | **否（未重现）** |
| FSC reverb | 1,240 | 0.877 | 0.861 | +0.017 | [−0.021, +0.054] | **否（未重现）** |
| FSC spatial（C5） | 720 | 0.815 | 0.799 | +0.017 | [−0.021, +0.050] | **否（未重现）** |

两种 LoRA 相对各自基线（target-any 配对差）：

| 条件 | 音频 LoRA − 原始 P2 | T2-LoRA − 原始 Whisper+T2 |
|---|---|---|
| SLURP C1 | +0.281 | +0.263 |
| SLURP C1-R | +0.276 | +0.252 |
| SLURP C5 | +0.260 | +0.258 |
| FSC clean | +0.061 | +0.098 |
| FSC reverb | +0.063 | +0.105 |

图（`docs/figures/`，由 `scripts/figures/make_figures.py` 从结果文件生成）：
- `fig1_error_ladder.png`：误差分解阶梯；
- `fig2_risk_coverage.png`：风险-覆盖率；
- `fig3_forest.png`：主要比较森林图（devel / test / FSC）。

## 局限
- 空间条件全部是 SMS-WSJ 模拟 RIR 卷积近讲语音再加白噪声，没有真实房间或真实阵列录音；只测了两人、SIR 0 dB。
- 询问方向在 C5 中是真值，在 c6-real 中是模拟的扇区 / 噪声，没有真实用户输入。
- 每种 LoRA 只训练一个种子、一个 checkpoint；只用一个 ASR（Whisper large-v3，未微调）。
- 只实现了"硬选一路"。融合方法 M3 和可靠性选流 M2 都没有实现。
- C3（未分离混合 + 方向说明）上音频 LoRA 更容易输出干扰者实体：0.128 → 0.195。
- FSC 的评分依赖冻结的 SLURP → FSC 映射表；映射表起草人曾看过 pilot 的部分输出（规则只依据标签定义和 FSC train 频次）。

## 文档索引
| 文件 | 内容 |
|---|---|
| `docs/slurp_test_report_v1.md` | **对外报告（论文底稿）**：设置、主要结果、FSC 跨数据集、风险、局限、后续工作 |
| `docs/slurp_test_freeze_v1.md` | SLURP test 冻结协议、审批决定、S0–S5 运行记录（作业号、commit、sha256） |
| `docs/fsc_extension_plan_v1.md` | FSC 扩充方案与冻结映射表（每条规则的依据和计数） |
| `docs/fsc_extension_results_v1.md` | FSC 扩充运行记录、结果、事后探索性分析 |
| `docs/lora_slu_plan_v2.md` / `docs/lora_v2_runs.md` | LoRA 训练计划与 devel 阶段全部运行记录 |
| `docs/mix_spec_v1.md` | 两人混合构造规则（mix_v1） |
| `docs/c6_real_v1.md` | 真实方向不确定性评估（扇区、噪声、拒答 A/B） |
| `docs/spatialnet_final_v1.md` | SpatialNet ep100 最终 checkpoint 与 devel 评估 |
| `docs/binding_score_v1.md` / `docs/scoring_rules.md` | 评分定义 |
| `docs/fsc_pilot_plan.md` | FSC 获取、审计与 64 条 pilot |
| `docs/plan_update_v2.md` | 主线修订与 M1/M2/M3 定义 |

## 复现
**环境**（`envs/` → `<SHARED_ROOT>/envs`）：
- `qwen_omni`：torch 2.6.0+cu124、transformers 4.52.4，用于原始 Qwen 和 Whisper；
- `qwen_omni_lora`：同上加 peft 0.15.2，用于 LoRA 和大部分 CPU 评分；
- `tfgridnet`：用于 SpatialNet 推理、选路、作图（matplotlib）；
- `sms_wsj`：rir_generator 0.3.0，用于直达声参考。
- Slurm：<SLURM_ACCOUNT> 账户；GPU 作业为 <GPU_PARTITION> 1×A100、4 CPU、48 GB、20 分钟。作业包装脚本为 `scripts/lora/slurp_test_job.sbatch`，工作区不干净就拒绝运行；环境变量 `EVAL_SPLIT` 取 test / fsc_ext。

**脚本顺序（SLURP test）**：
1. 划分：`scripts/slurp/make_slurp_split_v2.py` → `configs/slurp_split_v2_seed2026.json` 与 `data/slurp/manifests/split_v2_seed2026/*`；
2. 训练：
   - 混响训练数据：`scripts/lora/reverb_single.py --split train`；
   - 音频 LoRA：`scripts/lora/train_audio_lora.py`；
   - Whisper 转写：`scripts/lora/baseline_audio.py --mode whisper --items train_all`；
   - T2-LoRA：`scripts/lora/train_step2_lora.py`；
3. test 构造：
   - `scripts/slurp/make_mix_list_v1.py --v2 --split test --n 200`；
   - `scripts/sms_wsj/mix_slurp_v1.py`；
   - `scripts/lora/reverb_single.py --split test`；
   - `scripts/sms_wsj/direct_refs_mix_v1.py`；
4. 前端：`scripts/sms_wsj/frontend_snapshot.py --mode cv/mix`；`scripts/sms_wsj/select_final.py`；
5. 推理：
   - `scripts/lora/baseline_audio.py`（P2、Whisper + T2）；
   - `scripts/lora/eval_audio_lora_ckpt.py`；
   - `scripts/lora/eval_step2_lora.py`；
   - `scripts/qwen/qwen_omni_c0_devel428.py --split test`；
   - `scripts/qwen/qwen_omni_c3_p2.py`；
   - `scripts/lora/eval_audio_lora_mix.py`；
   - `scripts/qwen/qwen_omni_snapshot.py`；
   - 分片合并：`scripts/lora/merge_test_parts.py`；
6. 评分：`scripts/qwen/eval_final_routes.py`；`scripts/lora/eval_final_lora_checks.py`；`scripts/lora/score_slurp_test_v1.py`。

全部作业号和命令见 `docs/slurp_test_freeze_v1.md`；`scripts/lora/slurp_test_s0_check.py` 是 devel 等价核对。

**FSC**：
1. `scripts/fsc/ext_build_list.py`；
2. `ext_make_reverb.py`；
3. 混合 / 分离 / 选路沿用上面的脚本，`EVAL_SPLIT=fsc_ext`；
4. `ext_tasks.py`；
5. `ext_batch.py`；
6. `score_extension_v1.py`。

**关键文件 sha256**：
| 文件 | sha256 |
|---|---|
| Qwen2.5-Omni-7B | HF commit `ae9e1690` |
| Whisper large-v3 | HF revision `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| SpatialNet ep100 `best.pt` | `21caf0fc3b08cfc3b9860efbfeff9a756acecbe36101620378e47891edd76c10` |
| 音频 LoRA `lora_v2/step1/lr1e-4/ckpt_step440/adapter_model.safetensors` | `66d6a51d69a5d86f196819cff24ba49aa817aed92055460dd837bef6d0bed0cf` |
| T2-LoRA `lora_v2/step2/lr1e-4/ckpt_step660/adapter_model.safetensors` | `904039c1763676bad4f41b2a45c033e1891bf1e2ebdad4a7add0f50bacb4c68b` |
| SLURP 标注 `data/slurp/annotations/8eb1654/test.jsonl` | `fe3449af69b42fda7163345482556066c35a80f7e552a6d67e212a0e2f0783cc` |
| `configs/slurp_split_v2_seed2026.json` | 原始运行版 `eed16990a8f284c59393a7fed6fba2ec46b5dd2c3c215d1cb08e2f217a091d8c`；**公开版** `2ff8e1901d884d320ff276d578b0d2ba4858559888c35790d2a9ed28fb871ccc`（只差路径字段，路径已替换为占位符） |
| `configs/mix_v1_test_v1.json` | 原始运行版 `26c0fb57e79fa4442526c824cd1d2c79e558af727dc1c11f41e4dea26b02695f`；**公开版** `22283f4cb3fc09a20f32a7a5775a1689f10bee7374a5042ed49c3ee8439f91a5`（只差路径字段，路径已替换为占位符） |
| `configs/slurp_test_v1_s1_sha256.txt`（全部 test 输入文件） | 原始运行版 `d6475d2e27fc77cd466176759d9baba57a4db6ef4eca868bc09e0395241cd972`；**公开版** `ad8549a2b8a6c1284fdac60a27c2f4929d385651da6cb118f8f09fdc2915d342`（只差路径字段，路径已替换为占位符） |
| SLURP test 报告 `outputs/slurp_test_v1/score/report/report.json` | `e457aa3cbdd16a6ec68bfc9204671cda811d2b488ce983c540e074a4d994db8f` |
| FSC 映射表 `configs/fsc_mapping_v1.json` | `7a385e29c67bf9ac57ac034aa4149583b0d5432179cea71e9e1363d48f2f0ced` |
| FSC 报告 `outputs/fsc_ext_v1/score/report.json` | `dc694398b0ff00f22b4e7eccc6aa655b4e1c63da65fe089e21521299193218e5` |

**SLURP 划分清单**（split_v2_seed2026）：
- 运行时路径为 `data/slurp/manifests/split_v2_seed2026/`。`data` 是指向共享存储 `<SHARED_ROOT>/data` 的符号链接，git 无法跟踪链接后面的文件，所以仓库中保存的是**逐字节相同的副本** `manifests/slurp_split_v2_seed2026/`（2026-11-05 原文件，含原 `.csv.gz`）。
- 恢复方法：`cp manifests/slurp_split_v2_seed2026/* data/slurp/manifests/split_v2_seed2026/`。

| 文件 | sha256（文件本身） | sha256（gzip 解压后内容） |
|---|---|---|
| `train_recordings.csv` | `782f458940665b5010f2be5b0eaa497fe3ee52c5a3d7cbe2bd4df12368f125f9` | — |
| `devel_recordings.csv` | `08ebe28425aa311b89d1376d3e24cd1b15c3f6fe30489e1afa36aab6ebc21498` | — |
| `test_recordings.csv` | `81efdf51e7c6017b5840eb73a53158e3197b7eb9782c3a70a365b7629242df7b` | — |
| `train_candidate_pairs.csv.gz` | `d2515a16f4ca955a19177bdc0237e9399aa0f1c8a9f14be9f5a34ec3bd819f77` | `d3edcf7add7cacfc75748d5e8d1ecab7c10c58b19b8d01089df622c82bf1a90d` |
| `devel_candidate_pairs.csv.gz` | `6bf58c549b368cd6ecdc8aaf9d49014af8b9003a95cfe423d48e406ee8fe132f` | `cc2ca3f4c6ca4f0fa4a7c0c2eb531a699331464482979903c1ec3507f31bef37` |
| `test_candidate_pairs.csv.gz` | `383a32e69c04a49a47d62362e4606d42ad2a58554b249d1b764810e5b8f8d375` | `fd16f01c0d088fa2b2054e9d0f0e1e3cb48277f06c115e1c455fe8e615a01b50` |
| 输入：`results/slurp/headset_durations.csv`（录音时长表，重建划分的输入，已入库） | `0af9a40364573daa8651aaafc9351a4bff8f62daf4e43a6ab1dfd8eb955ee04e` | — |

**重建划分**：`python -I scripts/slurp/make_slurp_split_v2.py data/slurp/annotations/8eb1654 results/slurp/headset_durations.csv data/slurp/manifests/split_v2_seed2026`（**建议传入相对路径**，2 CPU、8 GB、约 40 分钟）。2026-11-13 已在 CPU 上重建核对过（记录见 `docs/slurp_test_freeze_v1.md` 末尾）：
- 三个 `*_recordings.csv` 逐字节相同；
- 三个 `.csv.gz` 的**文件** sha256 会因 gzip 文件头中的写入时间戳不同而不同，**以解压后内容的 sha256 为准**（重建版本解压后逐字节相同）；
- 脚本还会改写 `configs/slurp_split_v2_seed2026.json` 和 `results/slurp/split_v2_stats.json`。差异只在路径字段（传入绝对路径时）和浮点末位，**不影响清单内容**。

`data/`、`outputs/`、checkpoint 和音频都不在仓库中（见 `.gitignore`），所有路径都在共享存储上。

## 数据许可
- **SLURP 音频**为 CC BY-NC 4.0，仅限非商业使用。本项目由 SLURP 生成的衍生音频（混响、混合、分离输出）**不公开**。
- **FSC**：FSC 请从 Fluent.ai 官方渠道获取（官方数据页与许可见 `configs/fsc_pilot_pending.json`）；本项目只将 FSC 用于**非商业学术研究**，所用副本的再分发授权**尚未得到 Fluent.ai 确认**，发表前需确认。获取与审计见 `docs/fsc_pilot_plan.md`。衍生音频同样**不公开**；FSC official test 没有使用。
- **SMS-WSJ RIR**：只在本地使用，没有再分发。


## 路径占位符
公开版把集群相关的路径和设置替换成了下列占位符（代码、配置和文档中都是）。运行前请改成你自己的值。

| 占位符 | 含义 | 应替换为 |
|---|---|---|
| `<REPO_ROOT>` | 本仓库的绝对路径 | 你克隆本仓库的目录 |
| `<SHARED_ROOT>` | 项目数据根目录：数据（`data/`）、环境（`envs/`）、checkpoint（`checkpoints/`，含 HF 缓存 `checkpoints/hf`）所在的共享存储目录 | 你存放 SLURP 音频、生成数据、环境和权重的目录 |
| `<SEPARATION_DATA_ROOT>` | SMS-WSJ 数据根目录（含 `data/sms_wsj/sms_wsj.json` 和 `rirs/`） | 你的 SMS-WSJ 生成目录 |
| `<SEPARATION_REPO>` | 分离前端代码仓库（SpatialNet / TF-GridNet 训练代码和实验目录） | 你的分离前端代码目录 |
| `<SHARED_STORAGE>` | 共享存储的上一级目录 | 同上，按你的存储布局 |
| `<CONDA_ROOT>` | 个人 conda 安装目录（`tfgridnet` 环境所在） | 你的 conda 目录 |
| `<HOME>` | 用户家目录 | 你的家目录 |
| `<USER>` | 集群用户名 | 你的用户名 |
| `<SLURM_ACCOUNT>` | Slurm 账户（`-A` / `--account`） | 你的 Slurm 账户 |
| `<GPU_PARTITION>` / `<CPU_PARTITION>` / `<PARTITION>` | Slurm GPU / CPU 分区 | 你集群上的分区名 |
| `<NODE>` | 运行记录中的计算节点名（只出现在文档里） | 无需替换 |
| `<GROUP>`、`<PROJECT_GROUP>`、`<CLUSTER_DOMAIN>`、`<EMAIL>` | 集群组名、项目组名、集群域名、邮箱（只出现在文档里） | 无需替换 |
| `<FSC utterance>` | 文档中被删去的一句 FSC 原句 | 无需替换 |

## 许可与引用
- 作者：Yunqi Li。仓库：https://github.com/yunqiyunqili/audio-llm-target-speaker
- 代码与文档：MIT（`LICENSE`，Copyright (c) 2026 Yunqi Li）。数据与模型许可：`DATA_LICENSES.md`；引用：`CITATION.cff`。
- 注意：报告和冻结文档里记录的 `configs/slurp_split_v2_seed2026.json`、`configs/mix_v1_test_v1.json`、`configs/slurp_test_v1_s1_sha256.txt` 的 sha256 是原始运行版；公开版这三个文件只差路径字段，其 sha256 见上方"关键文件 sha256"表。
- 公开版中的集群路径、账户、分区、节点名已替换为占位符，如 `<SHARED_ROOT>`、`<REPO_ROOT>`、`<SLURM_ACCOUNT>`、`<GPU_PARTITION>`、`<NODE>`。运行脚本前，需要把这些占位符改成你自己的路径和集群设置。
- FSC 逐条样本清单（`configs/fsc_ext_v1_list.json`）没有包含在公开版中。可按 `scripts/fsc/ext_build_list.py` 的规则，从你自己获取的 FSC 数据重建。
