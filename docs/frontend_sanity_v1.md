# 前端流程核查 v1（frontend-sanity-v1；2026-11-05；方法和解释规则在运行前写定，看到结果后不修改）

目的：判断 frontend_mix_v1（作业 22014183）的结果偏低，是现有流程有 bug，还是短句或 SLURP 领域差异造成的。不运行 Qwen，不跑 C4/C5；不修改 mix_v1、已有前端结果或 frontend_mix_v1 的判定；不写 `data/mix_v1`。

共同设置：前端为 BLSTM+MVDR 和 FaSNet-TAC，checkpoint 只读，加载前核对 sha256 和配置。推理代码与 `scripts/sms_wsj/frontend_mix_v1.py` **逐行相同**：用 soundfile 读 float32 的 (T, 6) → 转成 (1, T, 6) → `T.normalize` → `model(x) × scale`。SI-SDR 用原 `tfgridnet/objectives.py` 中的 `sisdr` 和 `scores()`（PIT，两人平均）。cv_dev93 的 50 组用 `random.Random(2026).sample(sorted(cv_dev93 ids), 50)` 抽取，B 和 C 共用这 50 组。输出写到 `outputs/frontend_sanity_v1/<jobid>/`，只写指标和 JSON，**不写音频**。

## A. 推理流程核查（SMS-WSJ test_eval92，前 50 条）
- 数据：原评估使用的数据集（`sms_wsj_direct_v1`，按 `test_eval92.jsonl` 的顺序取前 50 条），参考是原评估用的 mic 0 直达声。
- 先核对：用 soundfile 直接读的混合信号，与原数据集 `__getitem__` 返回的混合信号完全相同。
- 用上面的推理代码和原 `scores()` 计算每条的 si_sdr_db、si_sdri_db，与原 `test_utterances.jsonl` 逐条比较。
- **规则**：如果任一前端满足以下任一条——每条差值的最大值 > 0.05 dB、平均差值 > 0.01 dB、PIT 排列有不一致——就判为**现有推理流程有问题**。

## B. 混合脚本核查（cv_dev93，50 组）
- **我方流程**：用 WSJ 原句（`original_source`，本来就是 8 kHz，所以跳过 16k→8k 这一步）和对应的 h_0、h_1，套用 `mix_slurp_v1` 的同一流程：rir_start、50 ms early、`extract_piece(offset − rir_start)` 对齐，偏移用官方 offset，长度用官方 T；每个声源缩放到 mic 0 early image 的有效帧 RMS = −30 dBFS（SIR 0 dB）；加白噪声，SNR 用官方值，噪声种子由 `np.random.RandomState(2026 + 组序号)` 产生。参考是我方流程算出的 early image（mic 0）。只在内存中生成。
- **官方对照**：官方 observation 和官方 early image（`speech_reverberation_early`，mic 0）；另外报告用官方直达声作参考的结果（`sms_wsj_direct_v1/cv_dev93`）。
- 指标：两人平均 SI-SDRi（PIT，early 参考）。另外报告官方的 SIR（log_weights）分布。
- **规则**：任一前端满足 |平均 SI-SDRi(我方流程) − 平均 SI-SDRi(官方)| > 2.0 dB，就判为**混合流程或电平规则有问题**。

## C. 时长核查（cv_dev93，同样 50 组，官方混合和官方参考）
- 截断长度：每组从 mix_v1 的 200 个 T_8k 中用 `np.random.default_rng(2026)` 抽一个 L（如果 L ≥ 原长度，就不截断，并记录下来）。
- 截断窗口：以 0.1 s（800 个采样点）为步长滑动，选两人有效帧重叠最多的位置（有效帧按官方 early image 的 mic 0 计算，256 点帧，峰值 −40 dB）；观测信号和参考信号截同一段。
- 比较：截断前后的两人平均 SI-SDRi（early 参考），以及 mix_v1 的平均值（作业 22014183：BLSTM+MVDR 2.90，FaSNet-TAC 0.47）。
- 记 gap = SI-SDRi(官方完整) − SI-SDRi(mix_v1)，drop = SI-SDRi(官方完整) − SI-SDRi(官方截断)。
- **规则**（每个前端分别判断）：drop ≥ 0.5 × gap，并且 gap > 2 dB 时，判为**短句是主要原因**；否则**短句不是主要原因**。gap ≤ 2 dB 时只报告，不判断。

## D. θ1/θ2 不对称核查（只读 22014183 的已有输出和 mix_v1）
- 用 `eval_frontend_mix_v1.py` 中的 `doa()` 估计方向，对象是：观测混合信号的 mic 0、oracle early image（A 和 B）、两个前端各自的两路输出。对每个对象，统计估计方向更接近 θ1 还是 θ2（按 |wrap(θ̂ − θ)| 比较，相等时记为平局）。
- 分组报告"更接近 θ1"的比例：按 s1/s2、按说话人 A/B（输出路按参考匹配归属）、按夹角分层。95% 置信区间：以组为单位 bootstrap，10,000 次，`numpy default_rng(2026)`。
- 结构检查：
  - 观测文件是 8 kHz、6 通道，形状 (T, 6)，T 与 manifest 一致；
  - 输出文件是 8 kHz、单通道、长度 T；
  - 通道顺序：逐通道与 early_A + early_B 的对应通道做相关，最大相关必须落在同一编号的通道上；
  - FaSNet-TAC 两路的能量比和接近静音的样本数（较弱一路的 RMS < 1e-4 × 较强一路）。
- 解释 FaSNet-TAC 目标路误差 77.9° 和干扰路误差 1.9° 的差异，以及 [30,60) 低于随机的原因：看两路方向分别落在哪里（θ1、θ2 或别处），并与观测混合信号的方向对比。
- **规则**：满足以下任一条，就判为**现有流程有 bug**：
  - 结构检查有任一项失败；
  - oracle early image 估计方向更接近自己位置（A 在 s1 是 θ1、在 s2 是 θ2；B 反过来）的比例 < 0.95。

## 结论（只能选一项）
按 A → B → D 的顺序检查：
1. A 或 D 判为 bug：**现有推理流程有 bug**，列出位置，不自行修复，mix_v1 的前端结果标记作废。
2. B 判为有问题：**混合脚本或电平有 bug**，同上处理。
3. A、B、D 都正常：**流程正常，主要是短句或 SLURP 领域差异**。由 C 的结果说明短句是不是主要原因；不是的话，就归为 SLURP 领域差异（包括说话风格、录音条件、电平分布等短句以外的因素）。
4. 任一项没能完成：**暂时无法判断**，说明还缺什么。

阈值都是经验值，不是显著性检验。

## 实现
- `scripts/sms_wsj/frontend_sanity_v1.py`（A、B、C；GPU；tfgridnet 环境；用 `--model` 指定前端）和 `.sbatch`。作业配置：<SLURM_ACCOUNT>、<GPU_PARTITION>、1 张 A100、4 CPU、32 GB、1 小时。
- `scripts/sms_wsj/frontend_sanity_d.py`（D；CPU；只读）。

## 结果（2026-11-05）
- 运行记录：
  - A/B/C：GPU 作业 22014492，<NODE>，13:05–13:07，退出码 0。作业记录的 commit 是 `d107701`，DIRTY = 1，那个未提交的文件是当时还没提交的 D 脚本，A/B/C 不依赖它。CPU dry-run 为 srun 22014488。
  - D 和最终汇总：srun 22014499（<CPU_PARTITION>），D 脚本的 commit 是 `18b4150`。
- 结果在 `outputs/frontend_sanity_v1/22014492/`：`sanity_{blstm_mvdr,fasnet_tac}.json`、`sanity_D_and_decision.json`、`sanity_D.log`。只有指标，没有写音频。

**A. 推理流程**（test_eval92 前 50 条）：两个前端逐条与原记录的最大差都是 0.0 dB，平均差 0.0 dB，PIT 排列没有不一致；用 soundfile 直接读的混合信号与原数据集加载结果完全相同。→ **没有问题**。

**B. 混合脚本**（cv_dev93 50 组，early 参考，两人平均 SI-SDRi）
| | 用 mix_v1 流程重新混合 | 官方混合 | 差值 | 官方混合，直达声参考 |
|---|---|---|---|---|
| BLSTM+MVDR | 6.62 | 6.94 | −0.33 | 4.17 |
| FaSNet-TAC | 1.43 | 1.39 | +0.04 | 15.05 |
- 混合信号的 SI-SDR：mix_v1 流程 −0.94，官方 −0.92。官方两人 log_weights 之差的中位数 1.65 dB，最大 4.59 dB。
- 两个前端的差值都 ≤ 2 dB → **没有问题**。

**C. 时长**（同 50 组，截到 1.7–4.3 s，取两人重叠最多的窗口）
| | 完整 → 截断（early） | 完整 → 截断（直达声） | gap（官方完整 − mix_v1） | drop | 判定 |
|---|---|---|---|---|---|
| BLSTM+MVDR | 6.94 → 5.45 | 4.17 → 3.45 | 4.05 | 1.50（占 gap 的 37%） | 短句不是主要原因 |
| FaSNet-TAC | 1.39 → 1.07 | 15.05 → 14.16 | 0.91 | 0.31 | gap ≤ 2 dB，只报告 |

**D. θ1/θ2 不对称**
- 结构检查：观测文件 200/200 正确，通道顺序 200/200 正确，两个前端的输出形状都是 200/200 正确。→ **通过**。
- oracle early image 判断自己所在位置：A 1.00，B 0.985（95% 置信区间 [0.965, 1.0]）。→ **通过**。
- 估计方向更接近 θ1 的比例：观测混合 0.48 [0.43, 0.54]；BLSTM+MVDR 的 A 路和 B 路都是 0.47；FaSNet-TAC 的 A 路 0.41 [0.34, 0.47]、B 路 0.40 [0.33, 0.47]。
- 按互换样本分开看，两路都跟着位置变：s1 中 A 路接近 θ1 的比例，BLSTM+MVDR 0.62、FaSNet-TAC 0.58；s2 中 B 路接近 θ1 的比例，BLSTM+MVDR 0.55、FaSNet-TAC 0.59。
- FaSNet-TAC 两路：
  - 被归为干扰路的那一路，有 100/200 落在 θ2 附近（±10°），所以干扰路误差小；
  - 被归为目标路的那一路，有 116/200 落在 θ1、θ2 都不靠近的地方，所以目标路误差大（中位数 77.9°）；
  - 两路 RMS 比的中位数 0.80，接近静音的样本 4 个。
- [30,60) 低于随机的原因：这一层有 35/50 的干扰路落在 θ2，同时有 35/50 的目标路没有明确方向；θ2 离 θ1 只有 30–60°，所以方向匹配倾向于选中干扰路。
- BLSTM+MVDR：200 个样本中有 147 个两路方向都落在同一侧（都靠近 θ1 或都靠近 θ2）。

## 结论（按规则只选一项）
**流程正常，主要是短句或 SLURP 领域差异。**
- 对 BLSTM+MVDR：短句只解释了 gap 的 37%，所以按规则归为 **SLURP 领域差异**。
- **这是排除其他原因后得出的判断，不是直接证明**：A、B、D 都没有发现问题，C 也不足以解释差距，于是归为领域差异；但"领域差异"本身没有被单独测量。
- 不在冻结选项里的另一个发现：FaSNet-TAC 是按直达声训练的，用 early 参考时在官方混合上也只有 1.39 dB，换成直达声参考则是 15.05 dB。所以 frontend_mix_v1 中 FaSNet-TAC 的 early 和完整参考结果，以及基于这两种参考的参考匹配，不能代表它的分离能力。后续评估改用直达声参考，见 `docs/frontend_mix_v1_eval_v2.md`。
- 维持不变：frontend_mix_v1 的 v1 结果和判定，以及 mix_v1 音频。
