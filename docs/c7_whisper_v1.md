# C7-W v1：Whisper 转写 + T2 抽取（c7-whisper-v1；2026-11-05；规则在运行前写定，看到结果后不修改）

## ASR 模型（待确认后下载）
- `openai/whisper-large-v3`，revision **`06f233fe06e710322aca913c1bc4249a0d71fce1`**（Hugging Face API 显示 lastModified 2024-08-12，不需要申请权限，license 为 apache-2.0；2026-11-05 只查询了元数据）。
- 只下载 fp16 单文件权重和必要的配置文件，用 `allow_patterns` 指定，共 11 个文件，**3,091,519,764 B（2.88 GiB）**：
  - `model.safetensors`：3,087,130,976 B，LFS sha256 为 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`；
  - 另外 10 个文件：config.json、generation_config.json、preprocessor_config.json、tokenizer.json、tokenizer_config.json、vocab.json、merges.txt、added_tokens.json、special_tokens_map.json、normalizer.json。
  - **不下载** fp32 分片、flax、pytorch_model.bin（整个仓库共 24.7 GB）。
- 目标路径：`HF_HOME = <SHARED_ROOT>/checkpoints/hf`，即 `hub/models--openai--whisper-large-v3/snapshots/06f233fe…/`。共享盘剩余 203 GB。
- 下载方式：与 Qwen 相同，用 `huggingface_hub.snapshot_download(repo_id, revision=<sha>, allow_patterns=[…])`。
  - 环境：tfgridnet 环境中已有的 huggingface_hub，不装任何东西；
  - 计划用 srun 在 CPU 节点运行：<SLURM_ACCOUNT>、<CPU_PARTITION>、2 CPU、4 GB、30 分钟；
  - 下载后与元数据逐一核对文件大小，并校验 model.safetensors 的 sha256。
- 加载：从本地 snapshot 目录加载，离线（`HF_HUB_OFFLINE=1`），`local_files_only=True`，与 Qwen 的做法相同，避免出现 22003339 之前遇到的 revision 问题。环境是 qwen_omni（transformers 4.52.4，已确认可以 import `WhisperForConditionalGeneration`），不装新包。

## 下载记录（2026-11-05，已获批准）
- srun 22016212（<CPU_PARTITION>，tfgridnet 环境中的 huggingface_hub 0.34.4），用时 9.7 s，只下载了 11 个文件。
- 核对结果：
  - 文件数 11，总大小 3,091,519,764 B，每个文件的大小都与元数据一致；
  - `model.safetensors` 的 sha256 为 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`，与 LFS 元数据一致（srun 22016215）。
- 磁盘：2.9 GB；共享盘剩余 203 GB。

## 条件
三组语音都是 16 kHz，按以下方式得到。只在内存中生成，不写音频：
| 组 | 音频 | 样本 |
|---|---|---|
| C7W-C1 | SLURP 原始 16 kHz 近讲录音（与 C1 相同） | mix_v1 中的 124 条录音；评估时映射到 200 个样本，方法与 R1 相同（取目标录音的输出，用这个样本自己的配对判定） |
| C7W-C4direct | 目标的 direct 参考 → `resample_poly(x, 2, 1)` → −30 dBFS（与 C4-direct 相同） | 200 |
| C7W-C5 | C5 选中的目标路 → 极性修正 → 8→16 kHz → −30 dBFS（与 C5 相同） | 200 |

- Whisper 解码：fp16，`WhisperProcessor` 提取特征，`generate(language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128)`，不输出时间戳，batch 1。
- 文本抽取：把 Whisper 转写 `.strip()` 后加一个换行，接冻结的 **T2 prompt**，用 Qwen2.5-Omni-7B 只输入文本。解码与 T2 相同：贪心，max_new_tokens 256。
- C7W-C5 依赖 oracle 方位，与 C5 相同。

## 指标
- 使用 `eval_c4_v1.score` 和 `summ`，不改动：只含目标、只含干扰、两者都不含、目标至少命中一个、目标全部命中、配对翻转准确率、M、U、解析失败、拒绝执行。
- 另外报告 Whisper 转写与目标句、与干扰句的 WER（normalize_v2）。

## 判定（经验值，不是显著性检验；a = 200 个样本上的"目标至少命中一个"；不修改已有结果）
- **已确认（2026-11-05）**：判定用 C7W-C5 与 C7W-C4direct 比较；P2 下的 C4-direct 只作参考。两者走的是同一条转写加文本链路：
  - |a(C7W-C5) − a(C7W-C4direct)| ≤ 0.05：判为"**转写和文本路线可能足够**"；
  - a(C7W-C4direct) − a(C7W-C5) ≥ 0.10：判为"**分离语音本身对转写也更难**"；
  - 其他情况：判为"**未能区分**"。
- 另外报告 C7W-C5 与 P2 下的 C4-direct（0.670）、C5（0.395）、C7 方案 A（0.255）的差值，只作参考，不参与判定。

## 作业（待确认）
- 一个 GPU 作业，把 Whisper（约 3 GB）和 Qwen（加载峰值约 21 GiB）放在同一张 A100 40GB 上。
- 配置：<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、48 GB、30 分钟。
- 预计用时：转写 524 条 + T2 抽取 524 条，估计 12–18 分钟，以实测为准。
- 脚本：`scripts/qwen/qwen_omni_c7w.py`（待写）。可以与 F1 的 Qwen 部分放在同一个作业里。

## 结果（2026-11-05；**不完整**）
- 运行记录：与 F1 在同一个 GPU 作业 22016221 里，**30 分钟时限到时超时**（17:02:44–17:32:56）。各部分用时：
  - Whisper：W_C1 39.2 s，W_C4direct 61.6 s，W_C5 60.3 s，全部完成；
  - T2：T_C1 221.4 s、T_C4direct 356.8 s 已完成；**T_C5 没有完成，也没有写出**。
- 这次 Qwen 每条推理约 1.9 s，以前约 1.1 s，所以作业比估计的慢。按规则没有改配置，也没有重交。
- 加载 70.2 s，加载时显存峰值 20.83 GiB。
- **C7-W 的判定无法计算**，因为缺少 C7W-C5 的 T2 输出。

### 已完成的条件
| 条件 | 只含目标 | 只含干扰 | 两者都不含 | 目标至少命中一个 | 目标全部命中 | 配对翻转准确率 | M | U | 解析失败 / 拒绝执行 |
|---|---|---|---|---|---|---|---|---|---|
| C7W-C1 | 0.565 | 0 | 0.435 | 0.650 | 0.565 | 0.39 | 189 | 9 | 0 / 0 |
| C7W-C4direct | 0.555 | 0.005 | 0.440 | 0.640 | 0.555 | 0.37 | 187 | 11 | 0 / 0 |
| C7W-C5 | — | — | — | — | — | — | — | — | 没有完成 |

### Whisper 转写（200 个样本口径，normalize_v2）
| 组 | 与目标句的 WER | 与干扰句的 WER | 与目标句 WER < 0.5 的条数 | 更接近目标 / 更接近干扰 |
|---|---|---|---|---|
| C1 | 0.075 | 1.045 | 189 | 196 / 0 |
| C4direct | 0.078 | 1.047 | 186 | 196 / 0 |
| C5 | **0.452** | 1.060 | **109** | 153 / 19 |

- 描述：在 C5 分离后的语音上，Whisper 与目标句的 WER 是 0.45，明显高于 direct 参考上的 0.078。这只是对转写这一步的描述，不是 C7-W 的判定。

## 补跑 T_C5（2026-11-05，已获批准；规则不变）
- 只读取作业 22016221 已写出的 `W_C5/raw.jsonl`，即 Whisper 对 C5 目标路的转写，**不重跑 Whisper**。用冻结的 T2 prompt 抽取，输入格式和解码设置与原作业完全相同：[转写.strip() + 换行, PROMPT_T2]，贪心，max_new_tokens 256。
- 入口：`scripts/qwen/qwen_omni_t_c5.py` 和 `.sbatch`，配置 <SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、48 GB、30 分钟。另外记录 commit、节点名、GPU 型号和每条耗时。
- 输出写到 `outputs/qwen_omni_c7w_f1_tc5/<jobid>/T_C5/`，不覆盖 22016221。
- 评估：新建 `outputs/qwen_omni_c7w_f1_combined/<jobid>/`，其中 F1、R1same、W_*、T_C1、T_C4direct 都是指向 22016221 的软链接，T_C5 指向补跑的结果；然后用 `eval_c7w_f1.py` 完整地重新评估一次。

## 补跑 T_C5 与完整评估（2026-11-05）
- T_C5 补跑：作业 22016732，节点 <NODE>，GPU 型号 **NVIDIA A100-PCIE-40GB**（驱动 610.43.02），20:58:04–21:03:15，退出码 0，MaxRSS 26.1 GB。
  - 作业记录的 commit 是 `eb85b2e`，没有未提交的依赖文件；输入 `W_C5/raw.jsonl` 的 sha256 为 `c6dee726…f705`。CPU dry-run 为 srun 22016722。
  - 加载 49.0 s；200 条共用时 207.6 s，**平均每条 1.04 s**，推理显存峰值 20.83 GiB。每条的耗时记在 `raw.jsonl` 的 `seconds` 字段。
  - 这次和 22016221 在同一个节点上，但每条快了很多：22016221 中 F1 和 R1-same 是音频输入的 P2，每条约 1.9 s；这里是纯文本的 T2，每条约 1.04 s。慢的原因没有单独查。
- 完整评估：在 `outputs/qwen_omni_c7w_f1_combined/22016732/` 下用软链接合并 22016221 的各条件和补跑的 T_C5，然后用 `eval_c7w_f1.py` 完整运行一次。结果在 `outputs/c7w_f1_eval/22016732_full/eval_c7w_f1.json`。F1 部分与之前单独计算的结果完全一致。

| 条件 | 只含目标 | 只含干扰 | 两者都不含 | 目标至少命中一个 | 目标全部命中 | 配对翻转准确率 | M | U | 解析失败 / 拒绝执行 |
|---|---|---|---|---|---|---|---|---|---|
| C7W-C1 | 0.565 | 0 | 0.435 | 0.650 | 0.565 | 0.39 | 189 | 9 | 0 / 0 |
| C7W-C4direct | 0.555 | 0.005 | 0.440 | **0.640** | 0.555 | 0.37 | 187 | 11 | 0 / 0 |
| **C7W-C5** | 0.290 | 0.005 | 0.705 | **0.340** | 0.290 | 0.13 | 109 | 87 | 3 / 3 |
| C5（P2，参考） | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 | 82 | 0 / 0 |

M 的口径：C7W-C5 按任一说话人统计；C7W-C1 和 C7W-C4direct 只看目标本人。

- **C7-W 判定（冻结规则）**：a(C7W-C4direct) − a(C7W-C5) = 0.640 − 0.340 = **0.30**，≥ 0.10 → **分离语音本身对转写也更难**。
- 只作参考（不参与判定）：C7W-C5 与 P2 下的 C5 相比低 0.055，与 P2 下的 C4-direct 相比低 0.33，与 C7 方案 A 相比高 0.085。
