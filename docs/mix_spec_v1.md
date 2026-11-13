# 混合配对规格 v1（mix_spec_v1；2026-11-05）

状态：配对过滤已完成；C3 方向说明已冻结（2026-11-05）；按下面第二版约束重抽清单（v2，取代第一版）；混合规则已写好，并做了不写音频的 dry-run。

## 数据来源
- 语音：只用 split_v2_seed2026 **devel** 中的录音（口径 A）。其中一部分句子来自官方 train，是 v2 规则划入 devel 的（规则见 `docs/provenance.md`）。**不使用 test。**
- RIR：只用 SMS-WSJ **cv_dev93**（开发集；8 kHz，6 个麦克风，每个样例有两个声源位置 h_0、h_1）。**不使用 test_eval92。**

## 配对过滤规则
基础条件：v2 devel 候选对，同性别，时长差 ≤ 1 s，每条录音最多用 3 次；D_T 和 D_I 都非空（定义见 `docs/binding_score_v1.md`，用 normalize_v2 处理后的值多重集合）。

**额外排除**：只要存在任一对 v ∈ D_T、u ∈ D_I 满足下面任一条，就排除这个配对：
1. **词干相同**：逐词取词干后的词序列相同。词干规则（一个简单规则，不是 Porter 词干算法）：
   - 先去掉词尾的 `'s`；
   - 长度 > 4 且以 `ies` 结尾时，改成 `y`；
   - 否则，长度 > 4 且以 s/x/z/ch/sh 加 `es` 结尾时，去掉 `es`；
   - 否则，长度 > 3、以 `s` 结尾且不以 `ss` 结尾时，去掉 `s`。
2. **包含**：一方的完整词序列作为连续子序列出现在另一方里（例如 "john" 和 "tara john"）。

实现：`scripts/slurp/devel_capacity_mixspec_v1.py`。

## 排除后的容量（srun 22011159，<CPU_PARTITION>；只做统计，不抽样，不生成音频）
- 基础条件下有 270 对，排除 22 对：21 对因为词干相同（都是 today 和 today's、tomorrow 和 tomorrows），1 对因为包含（john 和 tara john）。剩下 248 对，涉及 188 条录音。
- 每条录音最多用 3 次时：
  - 0/1 整数规划求出的精确最大值是 **185 对**（HiGHS 显示最优），比过滤前的 197 对少 12 对；
  - 贪心配对得到 182 对，这只是下限。
- 整数规划的一个最优解里：
  - 用到 186 条录音、11 个说话人（女 6、男 5）；
  - 按性别分，女 159 对、男 26 对；
  - 覆盖 27 个意图，减少最多的是 datetime_query（19 → 8 对）；
  - 按句子来源分，train+train 86 对、train+devel 79 对、devel+devel 20 对。
- 输出：`outputs/mix_spec_v1_capacity/devel/capacity_mixspec_v1.json`，包含全部 22 个被排除配对的原句。
- 已知限制：这个词干规则会把 "news" 和 "new" 判成同一个词干。本次被排除的 22 对里没有这种情况。

## 抽样清单 第一版（已作废）
- 第一版：`random.Random(2026)` 贪心抽样，结果 `configs/mix_v1_devel100.json`（srun 22012052）。作废原因：说话人和性别太集中，女声占 90 组，FE-250 出现在 50 组里。这份清单没有用来生成过音频。

## 抽样清单 第二版（约束在抽样前写定，2026-11-05）
**约束（必须全部满足，不放宽）：**
1. 只从上面过滤后剩下的 248 对中选（同性别、时长差 ≤ 1 s、D_T 和 D_I 都非空、没有共同词干或包含关系）。
2. 每条录音最多用 3 次。
3. **每位说话人最多出现在 30 组**（作为 A 或 B 都算）。
4. **男声至少 20 组**（配对都是同性别，所以一组要么是男声组，要么是女声组）。
5. 共 100 组。RIR 样例不重复；夹角分四层 [0,15)、[15,30)、[30,60)、[60,180]，每层 25 组。

**抽样方法（seed 2026）：**
- 把 248 对按 (file_a, file_b) 排序，用 `random.Random(2026)` 打乱，得到每对的随机排名 r（0 到 247）。
- 解一个 0/1 整数规划：在约束 2–4 和"正好 100 组"之下，最大化 Σ (248 − r)·x，也就是在可行解中尽量选随机顺序靠前的配对（scipy milp / HiGHS）。
- 如果这个规划不可行，就不放宽约束，而是报告最接近的可行结果，然后停止：
  - (a) 在约束 2–4 下最多能选多少组；
  - (b) 在 100 组、约束 2–3 下，能达到的最大男声组数和最小的说话人最大出现次数。
- 选出 100 对之后，再用同一个 rng 依次决定：A/B 方向、夹角分层标签的分配、每层内的 RIR 样例、偏移、噪声 SNR 和噪声种子。这些规则与第一版相同。
- **夹角分层只是为了诊断**：四层各 25 组，不代表 cv_dev93 的自然分布。在 cv_dev93 的 982 个样例里，[0,15) 有 78 个、[15,30) 96 个、[30,60) 155 个、[60,180] 653 个。
- θ1 = 位置 0（h_0）相对 mic 0 的方位，θ2 = 位置 1（h_1）的方位。方位定义见 `docs/conditions.md`：以阵列中心指向 mic 0 的方向为 0°，取值 (−180°, 180°]。
- 实现：`scripts/slurp/make_mix_list_v1.py`（v2 模式），输出到 `configs/mix_v1_devel100_v2.json`。

## 互换样本
- 每组生成两个样本：互换 1 为 A@θ1、B@θ2，目标是 A；互换 2 为 A@θ2、B@θ1，目标是 B。**两个样本询问的方向都是 θ1。**

## 偏移和对齐（与 SMS-WSJ scenario_map_fn 相同；代码已在 `repos/sms_wsj` commit f6b94bb 中核对）
- 偏移**跟着说话人走，不随方位变化**，两个互换样本相同。较长的一句从 0 开始，较短的一句在 [0, L_long − L_short] 中按 rng 均匀取一个整数（8 kHz 采样点）。混合长度 T = L_long，两句完全重叠。
- 对每个位置的 6 通道 RIR，rir_start = 各通道中第一个 |h| > 0.1·max|h| 的样本，取最小值（`get_rir_start_sample`）。early RIR = h[: rir_start + 400]（50 ms），其余置零。
- 卷积后用 `extract_piece(x, offset − rir_start, T)` 对齐，完整 image 和 early image 都这样处理。

## 电平
- 在每个互换样本里，把每个声源都缩放到：**mic 0 early image 的有效帧 RMS = −30 dBFS**。所以两个互换样本在 mic 0 early image 上都是 **SIR 0 dB**。
- 有效帧：把信号切成不重叠的 256 点帧（32 ms），帧功率 ≥ 峰值帧功率 − 40 dB 的帧算有效帧。
- 同一个增益用于该声源的全部 6 个通道，以及完整 image 和 early image。
- −30 dBFS 的核对（`scripts/sms_wsj/cv_dev93_level_stats.py`，srun 22013060，只读）：
  - cv_dev93 的 982 个混合信号，在 mic 0 上按同样的有效帧定义计算，有效帧 RMS 的中位数是 −33.84 dBFS（p5 为 −34.35，p95 为 −32.72）。
  - 每个说话人 early image 的中位数是 −35.18 dBFS。
  - −30 dBFS 与混合信号中位数相差 3.84 dB，在事先定的 6 dB 以内，所以**保留 −30 dBFS**。

## 噪声（核对 SMS-WSJ 后沿用）
- SMS-WSJ 的设置：6 通道高斯白噪声；SNR ~ U(20, 30) dB，参考功率是干净混合在所有通道和所有采样点上的平均功率（`get_white_noise_for_signal`）。
- 本规格的做法：每组用一个噪声种子和一个 SNR（都记在清单里），只生成**一份**噪声，两个互换样本使用**完全相同**的噪声数组。噪声的缩放参考是两个互换样本干净混合功率的平均值。因此单个样本实际的 SNR 会在组 SNR 附近略有偏差（dry-run 中为 19.7–30.1 dB）。

## 采样率转换（与 C1b 相同的 resample_poly 流程）
- SLURP 原始 16 kHz 音频先用 `resample_poly(x, 1, 2)` 降到 8 kHz，再和 8 kHz RIR 卷积并混合。
- 给 Qwen 的单通道输入：取 mic 0 的混合信号，用 `resample_poly(y, 2, 1)` 升到 16 kHz，截到 2T。
- 分离前端使用 6 通道 8 kHz 的混合信号。

## 输出（真实运行时；dry-run 不写）
- 每个（组, 互换）一个目录 `g{组号:03d}_s{1|2}/`，包含 `obs_6ch_8k.wav`、`mic0_16k.wav`、`early_A_6ch_8k.wav`、`early_B_6ch_8k.wav`（都是 float32），另有 `manifest.jsonl`。
- 实现：`scripts/sms_wsj/mix_slurp_v1.py`。

## C3 的目标方向说明（2026-11-05 冻结）
放在 P2 prompt 前面，作为音频之后的第一段文字；{theta} 为 θ1 四舍五入到整数度：
> The audio contains two overlapping speakers. Process only the speaker at azimuth {theta} degrees. Azimuth is measured from the array center, starting from the direction of microphone 0, counterclockwise positive, in the range -180 to 180 degrees.

- **正方向核对**（cv_dev93 样例 `0_4k6c0303_4k4c0319`）：
  - 坐标的第三维是高度（房间高 3.073 m，麦克风 z 在 1.550–1.563 m 之间，声源 z = 1.557 m）。
  - az = atan2(dy, dx) 在 x-y 平面上计算，az_rel = wrap(az − az_mic0)，实现见 `scripts/sms_wsj/sms_wsj_geometry_stats.py`。
  - 6 个麦克风的 az 依次是 125.1°、184.9°、244.9°、304.8°、4.9°、64.8°，按编号逆时针递增。
  - 结论：在"x-y-z 为右手系、z 轴向上"的前提下（这是图像法 RIR 的通常约定，SMS-WSJ 的代码没有明确写出），az_rel 为正就是从上方看逆时针，与说明文字一致。
- 两个互换样本的 θ1 相同，所以文字完全一样，只有音频不同。
- C3 输入的是 **mic 0 单通道**混合信号，音频本身不含可以用来判断方位的多通道信息。这是 C3 条件本来的设计。
