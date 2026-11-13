# C0 v1：SLURP 标准转写 → 冻结 T2（c0-v1；2026-11-06；**方案和作业配置，尚未提交，等批准**）

> - 目的：完全去掉"听"的问题，看文本理解的上限（`docs/conditions.md` 中的 C0）。
> - 本文档只写方案和配置。**没有提交作业，没有写推理脚本。** 不下载、不微调、不改环境、不动训练作业、不碰 test；不修改 mix_v1、P2/T2 prompt、slu-score-v2、binding-score-v1、已有结果和判定。

## 1. 输入
- 同一批 **124 条录音**（mix_v1 各组 A、B 的录音，与 C1 / P2、C7W-C1 相同）的 **SLURP 标准转写**：`configs/mix_v1_devel100_v2.json` 中每条录音的 `sentence`（SLURP 原文，小写、无标点，例如 "do i need to bring a coat today"）。不做任何改写。
- Prompt：**冻结的 T2**（`scripts/qwen/qwen_omni_c1_p2t2.py` 的 `PROMPT_T2`），原文不改。输入格式与 C7W 的 T2 完全相同：`build(processor, [sentence.strip() + '\n', PROMPT_T2])`（`qwen_omni_c6c7.build`），即"转写 + 换行 + T2"。
- 已有的相关结果（只作参考，不替代 C0）：作业 22011390 的 T2 是在 c1_ext 的 100 条录音的标准转写上跑的，其中 **50 条**与这 124 条重叠。C0 仍然在 124 条上**全部重跑**，不拼接旧输出；跑完后报告这 50 条与 22011390 的输出是否逐条相同（只作一致性观察）。

## 2. 模型与解码
- Qwen2.5-Omni-7B，commit `ae9e1690`（本地快照，离线），bf16，sdpa，**纯文本输入，talker 关闭**（`disable_talker()`，`return_audio=False`）。
- 解码与 T2 相同：贪心（`do_sample=False`），`thinker_max_new_tokens=256`。

## 3. 评分（不修改）
- **binding-score-v1**：
  - 按录音（与 `eval_c3_v1.py` 的 C1 相同：每条录音按它所在各组的配对计算，再对录音平均）——**这是与 C1(P2) = 0.694 比较用的主数**；
  - 映射到 200 个样本（与 `eval_c7w_f1.py` 的 C7W-C1 相同，`eval_c4_v1.score(single=True)`）。
- **slu-score-v2**（`rescore_slu_v2.score_item`）：意图严格准确率、只比值 P / R / F1、零实体数、解析失败数。
- **错误分类**：用 c1-error-taxonomy-v1 的规则（`docs/c1_error_taxonomy_v1.md`，不修改）做同样的分类；e / f 的判断用**标准转写本身**作为 transcript。由于输入就是标准转写，f 类（值不在 transcript 里）只可能来自金标值与转写文本不一致（如撇号、标注拼写），会单独列出。

## 4. 作业配置（未提交）
| 项目 | 值 |
|---|---|
| 脚本（批准后再写） | `scripts/qwen/qwen_omni_c0.py`、`scripts/qwen/qwen_omni_c0.sbatch`；评估 `scripts/qwen/eval_c0_v1.py` |
| 账户 / 分区 | <SLURM_ACCOUNT> / <GPU_PARTITION> |
| 资源 | 1 张 A100（`--gres=gpu:a100:1`），4 CPU，48 GB 内存 |
| 时限 | 30 分钟 |
| 环境 | `<SHARED_ROOT>/envs/qwen_omni`；`HF_HOME=$SHARED/checkpoints/hf`，`HF_HUB_OFFLINE=1`，`TRANSFORMERS_OFFLINE=1`（与已有作业相同） |
| 预计时长 | 加载约 50–70 s；推理 124 条 × 约 1.04 s ≈ 2.2 分钟；合计约 **4 分钟** |
| 输出 | `outputs/qwen_omni_c0/<jobid>/C0/raw.jsonl` 和 `config.json`（记录 commit、未提交的依赖文件、sha256、节点、GPU 型号、每条耗时）；日志 `outputs/logs/qwen_omni_c0_<jobid>.out`；评估结果 `outputs/c0_v1_eval/<jobid>/` |
| 新增磁盘 | 约 1 MB |
- 流程：先在 CPU 上做 dry-run（srun，只构造输入、不加载权重推理，检查 124 条和输入格式），工作区干净时再提交 GPU 作业。作业失败只报告原因，不改配置重交。

## 5. 运行前写定的判读（经验值，只用点估计）
记 Δ = C0 − C1(P2)，两者都用**按录音**的"目标至少命中一个"（C1(P2) = 0.6935，作业 22014089）。
1. Δ ≤ 0.05，**且**检查 A 中 b + c + d 占 P2 未命中实体 ≥ 50% → **"主要是标注格式问题"**；
2. Δ ≥ 0.15 → **"音频路径有明显损失"**；
3. 其他 → **"两者都有"**。
- 说明（事实，不是预测）：检查 A 已经跑完，P2 的 b + c + d 是 17 / 58 = **29.3%**，低于 50%。所以按这条规则，无论 C0 结果如何，**第 1 种判读都不会出现**；C0 只会落在第 2 或第 3 种。规则不因此修改。
- 判读只针对这 124 条 devel 录音、一次运行。

---

## 结果（2026-11-06；按上面事先写定的规则，结果出来后没有修改规则）
### 运行记录
- 批准后写了 `scripts/qwen/qwen_omni_c0.py`、`qwen_omni_c0.sbatch`、`eval_c0_v1.py`（commit `c8596ee`）。
- CPU dry-run：srun 22023431（<CPU_PARTITION>，<NODE>），124 条输入全部构造成功，`DRY_RUN_PASS`，没有未提交的依赖文件。
- GPU 作业 22023448：<SLURM_ACCOUNT> / <GPU_PARTITION>，1 张 A100（NVIDIA A100-PCIE-40GB，驱动 610.43.02），4 CPU，48 GB，30 分钟；节点 <NODE>，2026-11-06 14:35:04–14:39:39（4 min 35 s），退出码 0，MaxRSS 26.1 GB。作业记录的 commit 是 `c8596ee`，没有未提交的依赖文件。加载 49.6 s，推理 141.7 s（平均每条 1.13 s）。排队时间约 1 小时。
- 输出：`outputs/qwen_omni_c0/22023448/C0/raw.jsonl`（124 条）；评估：`envs/qwen_omni/bin/python -I -B scripts/qwen/eval_c0_v1.py ...`，在登录节点运行，只读 raw 输出，结果在 `outputs/c0_v1_eval/22023448/`。

### 主要结果
| | C1 / P2（音频） | C7W-C1（Whisper + T2） | **C0（标准转写 + T2）** |
|---|---|---|---|
| 目标至少命中一个（按录音） | 0.694 | 0.653 | **0.782** |
| 目标至少命中一个（200 个样本） | 0.695 | 0.650 | 0.810 |
| 目标全部命中（按录音） | 0.571 | — | 0.668 |
| 意图严格准确率 | 0.766 | — | 0.734 |
| 只比值 F1（slu-score-v2） | 0.642 | — | 0.731 |
- **C0 − C1(P2) = 0.782 − 0.694 = +0.089**（按录音，0.08871）。
- 其他：只比值 P / R 0.790 / 0.681；零实体 2 条；解析失败 1 条（`audio-1497270747`，标准转写只有一个词 "indie"，模型回复"无法提取命令的意思"，也算一次拒答）；200 个样本上"只含干扰"为 0。
- C1 / P2 的意图准确率和只比值 F1 来自 `docs/c3_results_v1.md`；C7W-C1 的这两项没有按同一口径在 124 条上算过，表中留空。

### 错误分类（c1-error-taxonomy-v1 规则不变；C0 的 e / f 用标准转写判断）
| 类 | C1 / P2 | Whisper + T2 | **C0** |
|---|---|---|---|
| 命中 / 未命中（共 144） | 86 / 58 | 81 / 63 | **98 / 46** |
| a. 解析失败 | 0 | 0 | 1（2.2%） |
| b. 写法不同 | 0 | 0 | 0 |
| c. 边界不同 | 11（19.0%） | 12（19.0%） | 11（23.9%） |
| d. 类型错 | 6（10.3%） | 6（9.5%） | 7（15.2%） |
| e. 漏抽 | 28（48.3%） | 30（47.6%） | **27（58.7%）** |
| f. 听错 | 13（22.4%） | 15（23.8%） | **0** |
- C0 的 27 个漏抽中，`date` 15 个；其余为 place_name 3、event_name 2、definition_word 2、personal_info 2，以及 news_topic、game_name、business_type 各 1。
- C0 的 c、d 类例子与 P2 基本相同，例如 `house_place "bathroom"` → `device_type "upstairs bathroom light"`，`podcast_descriptor "next podcast episode"` → `podcast_name "next podcast"`。
- f 类为 0：在输入就是标准转写时，"听错"消失；之前 f 类中撇号问题造成的个例（如 `applebees`）在这里也没有出现。

### 与 22011390 重叠的 50 条
- 50 条的输入 prompt（`prompt_text`）**逐条相同**，原始输出（`raw_output`）也**逐条相同**（50 / 50）。

### 判读（按 §5 事先写定的规则）
- Δ = +0.089；检查 A 中 P2 的 b + c + d 占未命中的 29.3%。
- 规则 1 不满足（b + c + d < 50%），规则 2 不满足（Δ < 0.15）→ **判读：两者都有**。
- 只针对这 124 条 devel 录音、一次运行；阈值为经验值，没有做显著性检验。

### 观察（只描述）
- 去掉"听"之后，未命中从 58 个降到 46 个，减少的 12 个大致对应 P2 的 13 个 f 类（听错）。
- 剩下的 46 个里，漏抽（27 个）以及边界或类型与 SLURP 标注不一致（18 个）在三种输入下数目都差不多。**即使输入是标准转写，T2 也有约 59% 的未命中是值在转写里却没有抽出**，所以这部分与"听"无关。
- C0 的意图准确率（0.734）比 C1 / P2（0.766）低，原因没有查。
