# FSC Pilot 准备记录（冻结前，不启动作业）

## 数据核查

官方 Fluent.ai 说明确认 FSC 是真实录音的英文 spoken-command 数据集：30,043 条、97 位说话人、16 kHz 单声道 WAV；每条 CSV 记录包含 speaker ID、音频路径、transcript 以及 action/object/location 三个 slot。官方 split 为 speaker-disjoint：train 77 speakers/23,132 utterances，valid 10/3,118，test 10/3,793。许可为仅学术研究的 CC BY-NC-ND 4.0；官方页面同时明确禁止商业用途，包括训练、测试、benchmark 和产品开发。

当前工作区没有 FSC 音频、CSV 或许可证文件，只有 SLURP 本地数据。已访问 Fluent.ai 官方页面和其授权的 Google Groups 入口；入口可达但当前未提供可直接下载的已授权文件，未使用镜像、抓取链接或绕过访问限制。因此“本地可访问性”仍为 **未获取**，不能提交推理作业。官方许可允许非商业学术用途，但禁止商业训练、测试和 benchmark；本项目用途符合非商业学术范围，仍须保留许可证和署名条款。获取后必须保存官方压缩包、CSV、LICENSE、下载日期和 SHA256。现有 SLURP train annotation 有 11,514 条语义记录；本地 manifest 含 `usrid`，可用于 speaker 审计。FSC 与 SLURP 的文本/语义/说话人重叠审计在 FSC 文件到位前只能报告方法和可验证范围，不能声称零重叠。

### 官方下载地址核查（2026-11-09）

官方 Google Groups 帖子提供的 `http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz` 经 HEAD 和 `Range: bytes=0-0` GET 检查均返回 HTTP 404；未发生重定向，因而没有 Content-Type 或 Content-Length。HTTPS 同端口返回 TLS version 错误，默认 HTTPS 端口也返回 404。未下载任何字节，未使用第三方镜像或替代链接；详细检查记录在 `configs/fsc_pilot_pending.json`。当前状态为 `blocked_official_url_404`，需 Fluent.ai 恢复该地址或提供作者明确授权的新地址后才能继续。

### 备用来源审计（2026-11-09）

（公开版：此处原为第三方备份来源的地址与获取说明，已删去。FSC 请从 Fluent.ai 官方渠道获取，授权状态见 `DATA_LICENSES.md`。）

（公开版：此处原为第三方备份来源的地址与获取说明，已删去。FSC 请从 Fluent.ai 官方渠道获取，授权状态见 `DATA_LICENSES.md`。）

（公开版：此处原为第三方备份来源的地址与获取说明，已删去。FSC 请从 Fluent.ai 官方渠道获取，授权状态见 `DATA_LICENSES.md`。）

### 获取后的 CPU 审计（2026-11-09）

用户确认仅用于非商业学术研究后获取 `fluent.tar.gz`（公开版不给出第三方来源；FSC 请从 Fluent.ai 官方渠道获取）。实际文件大小为 1,552,390,845 bytes，SHA256 `c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b`。解压前只检查 tar 成员，未执行任何归档内文件；归档包含官方 `Fluent Speech Commands Public License.pdf`，其 SHA256 为 `2714d08523da55400a57326efb4b42d7ea70b4fccaf3b6abba069a77695fb78d`。

CPU 核验通过：train/valid/test 分别为 23,132/3,118/3,793 条，speaker 分别为 77/10/10；共 30,043 个 WAV，无缺失或损坏文件，全部为 16 kHz、单声道、16-bit PCM。FSC valid 与 SLURP train 的规范化 exact transcript overlap 为 349 条录音（25 个 unique 文本值）；valid 的 token 3-gram Jaccard≥0.8 近似候选为 15 条，均需在结果解释中标记为潜在文本重叠。跨数据集 speaker ID 没有可链接身份，因此只能报告“命名空间独立、真实身份 overlap 不可验证”，不能宣称完全 speaker-independent。FSC 的 31 个 action/object/location triples 与 SLURP 的 91 个 intents/18 个 scenarios 不是同一标签体系；semantic overlap 只记录为需要人工映射审计，不能用标签数量相等推断独立性。

Pilot 已改为 valid-only 的 64 条冻结样本，覆盖 10 位 speaker（每位 6–7 条），覆盖全部 31 个 FSC intent triples；manifest SHA256 为 `64a695df7de69a93d4e5abea7d13a31b642b8539dd9e9a7301c393a6abe8c173`。审计和 manifest 位于 `outputs/fsc_pilot/audit.json` 与 `outputs/fsc_pilot/valid64_manifest.jsonl`，冻结配置位于 `outputs/fsc_pilot/pilot_config.json`。FSC test 和 SLURP final test 均未读取用于决策；尚未提交 GPU 作业。

### 空间输入与 smoke 状态

已按既定 RIR/SIR/SpatialNet ep100 配置生成 64 个双人空间化 directed trials（32 对交换 target/interferer），并生成 64 个单人混响 WAV。空间输入为 8 kHz、6-channel、无 clipping；混合 SIR 为 0 dB，实际 SNR 约 24.44–25.49 dB。CPU 预检确认 64 个空间目录、64 个混响文件、采样率、声道数、manifest 和哈希一致；FSC valid64 manifest 未改动。逐样本 overlap 标记在 `outputs/fsc_pilot/valid64_overlap.jsonl`。

SpatialNet ep100 smoke/full 输入检查作业 22057000（1 条 smoke）和 22057001（64 条）均 `COMPLETED/0:0`。原始 Qwen smoke 22057002 通过，耗时 2:11。Audio LoRA smoke 22057011 失败，ExitCode 1，日志显示 qwen 推理环境缺少 `peft` 模块；按照失败停止规则，未重试、未修改环境，未提交 T2 smoke 或完整 Pilot。当前 Pilot 状态为 **blocked_dependency_missing_peft**，需先补充/确认已批准的推理环境后才能继续；2 GPU-hours 预算未因失败而扩大。

## 重叠审计

获取 FSC 后，固定执行：

1. 以 Unicode 小写、标点归一化、数字和空白规范化生成 exact-text hash；报告 FSC train/valid/test 与 SLURP **训练集** 的 exact overlap。
2. 用 token 3-gram Jaccard、字符 5-gram、以及冻结的 sentence-transformer 余弦阈值（阈值在查看结果前写入配置）筛选近似文本；人工复核候选，不把相同 intent 当作文本泄漏。
3. 将 FSC speaker ID 与 SLURP `usrid` 视为不同数据源命名空间；只有在官方元数据提供可链接身份时才报告 speaker overlap，否则标记为“不可验证”，不宣称无重叠。
4. 审计只比较 SLURP train，不读取或修改 SLURP final test。所有排除项写入 manifest，审计脚本和 hash 一并保存。

## 冻结语义评分

FSC 原生目标表示固定为三元组 `{"action": ..., "object": ..., "location": ...}`。不把 31 个 intent 直接粗化成唯一主指标：

- `intent_exact`：三个 slot 全部相等；
- `intent_family`：只作为辅助指标，映射表固定为 action/object 的人工审计表，并同时报告映射前 exact 结果；
- `slot/entity correctness`：逐 slot exact、macro slot accuracy、slot precision/recall；
- `target-any`：目标说话人的任一 gold slot value 被正确输出；
- `speaker-binding`：双人条件下输出 slot 是否来自目标 speaker；
- `wrong-speaker-only`：只命中干扰 speaker 的 slot value；
- parse failure、拒答覆盖率和作答风险单独报告。

若 FSC slot 无法与 SLURP entity 类型建立一一对应，则只在标准化 `concept/value` 层比较，并将不可映射项从该指标分母剔除、另报覆盖率；禁止用宽松 intent-family 映射替代 exact slot 结果。

## Pilot 设计

选择 FSC **valid split 的 64 条 utterances、至少 8 位 speaker、speaker-disjoint** 的固定子集；抽样 seed=2026，按 31 intent 和 speaker 分层，manifest 在任何模型预测前冻结。FSC official test split 完全保留，不参与 Pilot 选择、阈值、模型或门控决策。四个条件各保留同一语义样本集合：

- 64 条单人干净；
- 同 64 条经固定 RIR 的单人混响；
- 64 条双人空间化 directed trials（32 对、每对交换一次 target/interferer）；
- 同一双人 trial 的 DOA ±10°/±20° 方向不确定性查询。

总计 256 个音频/查询 trial，三条冻结路线（原始 Qwen、Audio LoRA step 440、Whisper+T2-LoRA lr=1e-4 step 660）共 768 次推理。双人条件复用既有 SpatialNet ep100 的 RIR、采样率、SIR 分层和 route 逻辑；只替换输入语音，保存混合音频、DOA、route manifest 和 SHA256。DOA 选路先在不看模型输出的情况下核对 manifest 与实际选中 stream；方向查询使用既定 tau=16° 和冻结 C6 规则。

预算按一张 A100、最多 2 GPU-hours 预留（含模型加载和日志余量）；CPU 侧转码、RIR/manifest 和统计不计入 GPU。任何一项出现 OOM、输入/配置 hash 不一致、样本数不是 256、缺失率超过 0、重复作业、或预计超过 2 GPU-hours，立即停止后续依赖步骤并报告，不自行重试或改配置。

## 预测前冻结清单

数据版本/许可证/hash、speaker-disjoint manifest、exact/near-duplicate 阈值、slot 映射表、四条件音频 hash、RIR/DOA/SIR 参数、prompt、checkpoint provenance、随机种子、评分器版本、bootstrap B=2000 seed=2026、拒答阈值和停止条件全部写入 pilot config；完成核对后才可申请 GPU 作业。该准备阶段未启动 GPU、未训练、未读取 SLURP final test。

### 修复依赖后的冻结 smoke 验证（2026-11-09）

Audio LoRA smoke 22057011 的失败原因为使用了基础 `qwen_omni` 环境，该环境实际没有 `peft`；失败日志和输出均保留。核查此前 SLURP 成功作业后，未安装或升级任何包，改用已验证的 `<SHARED_ROOT>/envs/qwen_omni_lora` 环境：Python executable 为其 `bin/python`，`peft=0.15.2`、`transformers=4.52.4`、`torch=2.6.0+cu124`、`accelerate=1.7.0`、`safetensors=0.8.0`。两环境的 `sys.path` 和包位置已核对，原始模型、checkpoint、prompt、manifest、评分协议均未改变。

修复后的 Audio LoRA smoke 22057022 使用原始资源配置（1×A100、4 CPU、48G、20 min），`COMPLETED/0:0`，耗时 00:01:35，逐样本输出为 `outputs/fsc_pilot/smoke_audio_fixed.json`。随后 Whisper+T2-LoRA smoke 22057024 在同一兼容环境和冻结 `step2/lr=1e-4/step660` checkpoint 上运行，`COMPLETED/0:0`，耗时 00:00:48，输出为 `outputs/fsc_pilot/smoke_t2_fixed.json`。两个日志均保留。此阶段累计 GPU wall time（含失败作业）约 00:05:41，约 0.095 GPU-hours；按 2 GPU-hours 预算估算仍剩约 1.903 GPU-hours。未提交完整 FSC Pilot，等待单独批准。

### 正式 Pilot 预提交阻断（2026-11-09）

收到完整 valid64 Pilot 批准后，提交前核查确认冻结输入数量完整：valid64 manifest 64、overlap manifest 64、单人混响 64、spatial_audio_real 64、SpatialNet ep100 输出 64；对应 manifest 哈希分别为 `64a695df7de69a93d4e5abea7d13a31b642b8539dd9e9a7301c393a6abe8c173`、`cf80c13c6f34dda1d3db3228de7b78dddc22ed1ad2d0f7822bef5e9ef58b80c7`、`825427d43554cf519bcefb442e2e67f5620029f5e31b39cc8453c7af98e99c9f` 和 `7dc146cc02b26cd5e4a646bb5d5efc7960c3be3417e2b8a6d08a311cf3bdad23`。队列中无 FSC 作业，重复作业检查通过。

但 FSC 专用的 DOA 选路/selected-stream manifest 未生成或未落地；`outputs/fsc_pilot` 中不存在 selection、DOA 或 route 文件。其他实验目录的选择结果不属于 FSC，不能替代冻结的 FSC 路由。由于双人和方向不确定性条件依赖该文件，按协议停止，未提交任何正式 Pilot GPU 作业，也未生成预测输出。此前累计 GPU wall time 仍为约 00:05:41（0.095 GPU-hours），剩余约 1.905 GPU-hours。需先补齐并冻结 FSC 路由 manifest，重新完成 CPU 输入/哈希预检后，才能提交 Pilot。

### FSC DOA / selected-stream Route Manifest（2026-11-09）

按已冻结的 SLURP C5/C6 实现生成 FSC 专用 CPU route manifest：零时延 mic0 极性修正、PHAT 加权 360° DOA 网格估计、与冻结方向查询的最近角度选择，平局选择 stream 0。选路只读取 FSC 6-channel observation、两个 SpatialNet ep100 输出、阵列几何和 `theta_question`；未读取 speaker、transcript、intent 或 entity。代码 provenance 为 `scripts/sms_wsj/eval_frontend_mix_v1.py` SHA256 `665c9d64228b66a45986ae7bf73ee46b73ca11ff9436fc163b99b795349b7acb`，`eval_frontend_mix_v1_v3.py` SHA256 `046c376c7774aa08a2a89a780cf842d14982b42610aed369076398fdcce79610`，生成脚本 SHA256 `1160cbf5e073319db56df5a8f3225d44f71c95cd51332edb49546c6b15b35171`。

CPU 预检生成 64 条唯一 primary route 记录、320 条方向不确定性记录（每条 `delta=-20,-10,0,+10,+20°`），每条记录含 sample/pair/swap、目标方向、两路 DOA、selected stream、音频路径、输入/输出 SHA256、极性修正、规则和代码 hash。64 条输入与输出均存在，输入/输出 hash 全部一致，pair-swap 元数据完整；32 对中 1 对的 selected stream 发生互补（互补率 0.03125），未用该诊断结果调节选路。

仅用于事后诊断的 reference matching 显示 primary selected-stream route accuracy 为 0.953125（61/64）；选中 stream 的平均/中位 DOA 误差为 0.6243°/0.4866°，最近 stream 的 16° 阈值覆盖率为 1.0。方向不确定性记录的拒答/覆盖元数据使用冻结 `tau=16°`，未根据 FSC 语义输出修改。详细逐样本文件为 `outputs/fsc_pilot/route_manifest/route_manifest.jsonl`、`direction_uncertainty_manifest.jsonl` 和 `diagnostics.jsonl`，汇总为 `summary.json`。本阶段未提交 GPU 作业；此前累计 GPU wall time 仍约 0.095 GPU-hours，剩余约 1.905 GPU-hours。

### Pair-swap / source-level route 只读诊断（2026-11-09）

FSC 每个 pair 由同一对源语音构成：swap 1 将 A 置于位置 0、B 置于位置 1 且 target=A；swap 2 将同一 A/B 交换物理位置（A 到位置 1、B 到位置 0）且 target=B。两次的目标方向查询 `theta_question` 相同，target/interferer source identity 互换。SpatialNet 的 stream 0/1 不保证跨 swap 保持 source identity，因此 stream index 不应要求互补。

只读 source matching 使用已有 early_A/early_B reference 对两个输出做事后 permutation matching，未反馈到 route manifest 或选路。32 对中 source-level 结果为：30 对两条均正确、2 对一条正确、0 对两条均错误（64 条中 62/64=0.96875）。异常 pair 为 g000（g000_s1 误选 B，g000_s2 正确）和 g011（g011_s1 正确，g011_s2 误选 A）；两对均确认 target identity 发生 A/B 交换、方向查询相同、无标签或 pair 构造错误。stream index 的互补率仅 1/32 是 permutation 表象，不能作为 source-level 错误率。

此前 61/64（0.953125）来自逐 stream 与目标 reference 的 legacy target-only matching；其 pair 统计为 29 对两条正确、3 对一条正确、0 对两条错误，额外的 g019_s1 属于两路输出相关性都偏弱时的 assignment 定义差异。完整 source permutation matching 将 g019 pair 归为两条正确；两种定义都保留，未修改冻结 route manifest 或评分协议。

0.6243° 平均 DOA 误差是 64 条 primary route 中“实际 selected stream 的 DOA 与目标方向查询”的平均绝对圆周误差；由于 selected stream 本身按最近 DOA 选择，它数值上等于 nearest-stream 误差，并包含 legacy 定义下的 3 条错路样本（其均值 1.3056°；其余样本均值 0.6023°）。`tau=16°` 的 100% 只是最近估计 DOA 到查询方向的阈值覆盖，不是正确说话人覆盖率。

诊断文件为 `outputs/fsc_pilot/route_manifest/pair_diagnostic.json`，脚本为 `scripts/fsc/diagnose_route_pairs.py`。未使用 FSC test、SLURP final test，未提交 GPU 作业；route manifest、算法、阈值和哈希均未改变。

### 正式 Pilot 第一阶段阻断（2026-11-09）

正式 Pilot 原始 Qwen 阶段作业 22059111 在第 3 个任务前失败（`FAILED/1:0`，00:02:11）。冻结任务清单中的 clean/reverb 输入为 16 kHz，但 Route Manifest 的 selected SpatialNet `out{k}.wav` 为 8 kHz；`qwen_pilot_batch.py` 按既定模型输入协议要求 16 kHz，因此在输入哈希/采样率断言处停止。仅前 2 条任务写入 `outputs/fsc_pilot/pred_base/predictions.jsonl`，阶段输出不完整，未提交 Audio LoRA 或 Whisper+T2-LoRA 阶段。未重采样、未修改 route manifest、任务清单、模型、prompt 或评分协议，也未自动重试。

本次失败新增 GPU wall time 00:02:11（约 0.0364 GPU-hours）；此前约 0.095 GPU-hours，累计约 0.131 GPU-hours，剩余约 1.869 GPU-hours。失败记录和部分输出保留，等待明确处理 8 kHz SpatialNet selected output 与冻结 16 kHz Qwen 输入之间的协议不一致后再继续。

## 预处理修复与重新冻结（2026-11-11，CPU / 登录节点，无作业）

### 批准记录
- **FSC valid64 pilot 由项目负责人于 2026-11-09 批准**（仅用于非商业学术研究）。此前本文档只写"用户确认""收到批准"，没有写批准人和时间，此处补记。
- 2026-11-11 项目负责人批准本节的两项预处理修复（只改 FSC 脚本，不改 SLURP 代码）、作废旧作业、取消跟踪 outputs 文件；FSC 的 GPU 冒烟测试和试跑**等 SLURP test 冻结文档审批后再提交**。

### 发现的两项不一致（2026-11-11 只读核对）
1. **单人混响**：旧 `prepare_spatial_inputs.py` 把 16 kHz 语音直接与 8 kHz RIR 卷积，再 `resample_poly(z,2,1)` 后按 16 kHz 写出并截断到原长（时间尺度错误：内容约为前半段、半速）；没有 rir_start 对齐；音量按峰值对齐，不是 −30 dBFS。与 SLURP C1-R 不一致。
2. **P2 路线多做一次 chain8**：未提交的 `qwen_pilot_batch.py` 在 base / audio_lora 路线调用 `build(qp, PROMPT_P2, chain8(wav))`：干净与混响音频被 chain8（×2 上采样 + 电平），空间条件 chain8 两次。T2 路线只做一次，正确。

### 修复
- 新 `scripts/fsc/make_reverb16k.py`：**直接调用 `scripts/lora/reverb_single.reverb()`**（不修改；绝对路径使其 `SLURP / file` 解析到 FSC 文件）。RIR = 本 pilot 对所用 mix_v1 组的 **cv_dev93** 样例（`configs/mix_v1_devel100_v2.json` groups[:32]，与 `spatial_list.json` 一致），位置 h_{pilot_index mod 2}，mic 0；16-bit PCM 写出（与 reverb_single.py 相同）。旧目录改名为 `outputs/fsc_pilot/reverb16k_void_20261109/` 保留，未删除。新 `reverb16k/manifest.jsonl` sha256 `e22b8da1bac6a0fb63031d36c29b2f6b663efe679b2b0930a1c7881ce911a2d2`；64 条，无削波，峰值最大 0.282。
- `prepare_spatial_inputs.py` 删除错误的混响生成部分（只保留 `spatial_list.json` 的生成逻辑，本次未重跑）。
- 新 `scripts/fsc/pilot_audio.py`：`prepare()` 是每个条件唯一的预处理路径，`qwen_pilot_batch.py` 改为调用它，P2 路线不再额外调用 chain8；输出记录 `n_chain8`。
- 新 `scripts/fsc/check_pilot_preprocess.py`：逐条件断言，结果 `outputs/fsc_pilot/preprocess_check.json`，**PREPROCESS_CHECK_PASS**。

### 逐条件断言结果（256 条，使用推理时同一个 `prepare()`）

| FSC 条件 | 对应 SLURP | 断言 | 结果 |
|---|---|---|---|
| clean（64） | C1 | 16 kHz；与原始 FSC 文件逐样本相同（不调音量）；chain8 0 次 | 通过 |
| reverb（64） | C1-R | 16 kHz / PCM_16；与 `reverb_single.reverb()` 重算的最大差 3.05e-5（≤ 1/32768，16-bit 量化）；有效帧电平 −30.0000 ± 0.0001 dBFS；chain8 0 次 | 通过 |
| spatial（64） | C5 | 输入 8 kHz；输出与手写一次 `level(resample_poly(sign·x,2,1)[:2n])` 逐样本相同；长度 = 2n；−30 dBFS；极性按 route manifest 的 `flipped[selected_stream]` 乘一次（64 条均为 −1，与 SLURP ep100 极性修正率 1.00 一致）；chain8 1 次 | 通过 |
| direction_uncertainty（64） | C5 链路（C6δ） | 同上；音频路径和选中流与 direction_uncertainty_manifest 一致 | 通过 |

- 参照核对：`reverb_single.reverb()` 重算 3 条 SLURP devel C1-R 文件，最大差 3.05e-5（16-bit 量化），确认调用的就是 C1-R 链路。
- **长度说明**：C1-R 链路输出长度为 2·ceil(n/2)。SLURP devel 428 条输入全是偶数长度，所以输出与输入等长；FSC 有 **45 / 64** 条输入为奇数长度，混响输出比干净音频多 1 个样本（1/16000 s）。这是链路本身的确定行为，未修改链路。

### 方向条件的定位
- FSC 方向条件（固定偏移 ±10°、±20°）**对应 SLURP 的 C6δ，只作敏感性分析**，不是真实 C6（c6-real-v1），不参与任何门控或判读。每个偏移 **16 条**（按 pilot_index mod 4 分配），SLURP C6δ 每个偏移是全部 200 条，两者不能直接比较样本量。拒答元数据使用 C6δ ep100 的 τ = 16°（c6-real-v1 的 τ = 17° 不适用于本条件）。

### 新冻结任务清单
- `outputs/fsc_pilot/pilot_tasks.jsonl`（不入库）：sha256 **`9f9cc65db6ca3b81f1f208a8ee4b9a1d258fcbd265fa9ed923d35af850fd9e36`**，256 条：clean 64、reverb 64、spatial 64、direction_uncertainty 64（−20° / −10° / +10° / +20° 各 16）。由 `scripts/fsc/build_pilot_tasks.py`（逻辑未改）生成。
- 旧清单作废：sha256 `bdd262745bd7446b49e5e4e801104e7b10fb71839a65bdd818b228b2c002e69b`，副本保留为 `pilot_tasks_void_20261109.jsonl`。
- `outputs/fsc_pilot/pilot_config.json` 中的状态和混响记录仍是 2026-11-09 的版本，已过时；以本节为准。

### 作废的作业
| 作业 | 内容 | 作废原因 |
|---|---|---|
| 22057002 | 原始 Qwen smoke | 输入为旧的错误混响 `reverb16k/000.wav`，且 `qwen_smoke.py` 又对它执行 `chain8()`（P2 路线多一次 ×2 上采样 + 电平）；输出 "Set the alarm for 7 am tomorrow"，与原句 "<FSC utterance>" 无关 |
| 22057022 | Audio LoRA smoke | 同一错误混响输入 + 多余 chain8；输出 "play radio" |
| 22057024 | Whisper + T2-LoRA smoke | 同一错误混响输入（T2 路线没有多余 chain8）；Whisper 转写为 " Thank you."，说明输入音频已不可懂 |
| 22059111 | 正式 pilot 原始 Qwen 阶段（FAILED，2 条输出） | 使用旧清单（含错误混响）和未提交的 `qwen_pilot_batch.py` 旧版本；在 8 kHz 断言处停止；输出 `pred_base/` 不完整 |
- 这些作业的 GPU 时间（约 0.131 GPU 小时）仍计入 FSC 的 2 GPU 小时预算，剩余约 1.869 GPU 小时。

### outputs 文件改为不跟踪
- `outputs/` 在 `.gitignore` 中，但以下 11 个文件曾被强制加入跟踪。2026-11-11 用 `git rm --cached` 取消跟踪，磁盘文件保留不变：

| 文件 | 字节 | sha256 |
|---|---:|---|
| outputs/fsc_pilot/audit.json | 9813 | `cc619fae7e039f09dce2ba55fc6e587c7719b5dd70c22d86b1d6a028dba36643` |
| outputs/fsc_pilot/pilot_config.json | 5745 | `031276d1e8ddba7cdc8afd2e7fa0604f9b0e877a0418ee1c1e588b13cf7f7d3c` |
| outputs/fsc_pilot/pred_base/predictions.jsonl | 1177 | `a343f2e84f81f81d359d3bdfcd05a13fb7838aa7456bd0e85338514984c13116` |
| outputs/fsc_pilot/route_manifest/diagnostics.jsonl | 28741 | `03e789efd483d21694980a8cdf116cf3e359fcdc869a3475cb17596c3d75a652` |
| outputs/fsc_pilot/route_manifest/direction_uncertainty_manifest.jsonl | 359557 | `7d8ed8364e6e08435eec5c4685cfa61363c6d20adf00fd1ff851ed1351474aef` |
| outputs/fsc_pilot/route_manifest/pair_diagnostic.json | 66128 | `e1e57a6cf13145e58a3bf824cf15fd7d67691d2edd2d4dc2b678989c19e4d838` |
| outputs/fsc_pilot/route_manifest/route_manifest.jsonl | 81052 | `de361ee54ec8772b89237b3579388e900e348ab5d168cbcc74d339d56402522e` |
| outputs/fsc_pilot/route_manifest/summary.json | 2376 | `91d86e712c058f366a10121bf9cdf57f520fd752865ec588a920c9141acd5327` |
| outputs/fsc_pilot/spatial_list.json | 41532 | `e114f68eb4bea7dc99aa42be25baab9173d9a5b312a433df74f2baeaee0fa903` |
| outputs/fsc_pilot/valid64_manifest.jsonl | 22861 | `64a695df7de69a93d4e5abea7d13a31b642b8539dd9e9a7301c393a6abe8c173` |
| outputs/fsc_pilot/valid64_overlap.jsonl | 34997 | `cf80c13c6f34dda1d3db3228de7b78dddc22ed1ad2d0f7822bef5e9ef58b80c7` |

- 新生成文件：`reverb16k/manifest.jsonl` `e22b8da1…a2d2`；`preprocess_check.json` sha256 见提交时记录（`check_pilot_preprocess.py` 可重算）。
- `scripts/fsc/qwen_smoke.py` 仍含多余 chain8，本次未修改；今后的冒烟测试改用 `qwen_pilot_batch.py --limit N`（走同一个 `prepare()`），不再使用 `qwen_smoke.py`。

### 补记（2026-11-11，项目负责人要求）
- **混响长度多 1 个样本**（再次明确记录）：reverb 条件按 SLURP C1-R 链路（`reverb_single.reverb()`）生成，输出长度为 2·ceil(n/2)。FSC valid64 中 **45 / 64** 条原始录音为奇数长度，其混响输出比干净音频多 1 个样本（1/16000 s，末尾）；19 条偶数长度输入等长。SLURP devel 输入全为偶数长度，所以 SLURP 上从未出现。这是链路本身的确定行为，没有修改链路、没有截断；`check_pilot_preprocess.py` 按该长度规则断言（`n_len_plus1_odd_input = 45`）。
- `scripts/fsc/qwen_smoke.py` 开头已注明弃用并直接退出；冒烟测试只能用 `qwen_pilot_batch.py --limit N`。
- 项目负责人 2026-11-11 决定：**不做冒烟测试**，直接提交 256 × 3 正式运行（base、audio_lora、t2_lora 各一个作业；<SLURM_ACCOUNT> / <GPU_PARTITION>，1 × A100，4 CPU，48 GB，20 分钟；`envs/qwen_omni_lora`）。提交脚本 `scripts/fsc/qwen_pilot_batch.sbatch`，输出 `outputs/fsc_pilot/pred_v2_<route>_<jobid>/`；旧 `pred_base/`（22059111）保留不动、作废。运行结果与核对记录见下一节。

### 正式运行 v2（2026-11-11，256 × 3，无冒烟）
提交前工作区干净，三个作业的日志都记录了 `GIT_COMMIT 28b1337`、`envs/qwen_omni_lora` python，以及 tasks sha256 校验 OK。

| 路线 | 作业 | sacct | 用时 | 条数 / 各条件 | raw_output 可解析 JSON（含 intent+entities） |
|---|---|---|---|---|---|
| base | 22076114 | COMPLETED 0:0 | 00:08:07 | 256 / 64×4 | 256 / 256 |
| audio_lora | 22076115 | COMPLETED 0:0 | 00:11:54 | 256 / 64×4 | 256 / 256 |
| t2_lora | 22076116 | COMPLETED 0:0 | 00:12:14 | 256 / 64×4 | 256 / 256 |

- 逐条核对：index、condition、pilot_index、audio_sha256 与任务清单一致；route、checkpoint、prompt 正确；n_chain8 为 clean/reverb 0、spatial/direction 1；只有 t2_lora 有 asr。
- 人工抽查：每个条件抽 2 条（`random.Random(2026)`），抽中 clean #15/#40、reverb #13/#28、spatial #53/#62、direction #30(+10°)/#56(−20°)，三条路线共 24 条。模型 transcript 和 Whisper 转写都与原句对得上。唯一的偏差是 clean #40 Whisper 把 "Turn down" 转成 "Countdown"，属于 ASR 本身错误，不是管线问题。
- GPU：本次 0.54 GPU h；FSC 累计约 0.67 / 2 GPU h。
- 尚未评分：FSC 标签到 SLURP intent/slot 的映射需按计划另行进行。
- 输出 sha256（`outputs/fsc_pilot/`，不入库）：`pred_v2_base_22076114/predictions.jsonl` 825ddd5b…c9be；`pred_v2_audio_lora_22076115/predictions.jsonl` 37b5c7b7…b28f；`pred_v2_t2_lora_22076116/predictions.jsonl` 63b09d59…d4d0。

### 评分（2026-11-12，CPU，`scripts/fsc/score_pilot.py`；按运行前冻结的规则，未修改、未补充映射）
- 输入：三个 v2 正式运行的 `predictions.jsonl`（sha256 见上节）；任务清单 sha256 `9f9cc65d…9e36`；映射文件 `configs/fsc_semantic_mapping.json`（2026-11-09 冻结，未改）。输出 `outputs/fsc_pilot/score_v2/fsc_pilot_scores.json`（sha256 `c607d2e6eb091f98a91fa23f24188a8fa54b0ec37482a8287668adba4ac462cf`）和 `per_item.jsonl`（sha256 `9b092d7307fb1bafac94121dc700209b50f67a7d132e11b33bdb4c4e9c39d231`）。同一脚本重跑到 `score_v2_r2/`，两个文件逐字节相同。
- **映射覆盖**：冻结文件只有规则，**没有**经审计的 SLURP 输出 → FSC 三元组映射表，也没有 FSC slot → SLURP 实体类型的对应。因此：
  - `intent_exact`（三个 slot 全对）**无法计算**（覆盖 0/256），`intent_family` 未计算；没有用宽松映射替代。
  - 其余指标都在冻结规则允许的"标准化 value 层"计算：金标 slot 值（去掉 `none`，`normalize_v2`）与模型输出实体值做整值精确匹配，不看类型。
  - action 值（activate / deactivate / increase / decrease / bring / change language）在 SLURP 格式里是意图而不是实体，几乎从不作为实体值出现，所以 action 召回接近 0。这是映射缺失造成的，**不是模型结论**。
  - 输出覆盖率（模型输出的实体值中等于某个 FSC slot 值的比例）：clean 0.68–0.83、spatial 0.59–0.75，见下表"覆盖"列；其余输出值（例如 `washroom lights`、`english`）在冻结规则下不计分。
- 双人条件的 binding 用 binding-score-v1 `judge`（D_T = 目标 − 干扰者），2/64 条两人 slot 值无差别，从 binding 指标中剔除。"只命中目标"要求 D_T 全中，而 D_T 含 action，所以该类别普遍偏低；"只命中干扰者"即 wrong-speaker-only。
- 方向不确定条件：作答按 route manifest 的 `within_tau`（冻结 C6 τ=16°）计，覆盖率 0.672（−20° 0.25、−10° 1.00、+10° 1.00、+20° 0.44）。
- 置信区间仅作描述：B=2000，seed 2026；单人条件按 utterance 重抽样，双人条件按 pair 重抽样。

| 路线 | 条件 | target-any [95%] | object / location 召回 | 值 P / R | 覆盖 | 只命中干扰者 | 转写 WER | 零实体 |
|---|---|---|---|---|---|---|---|---|
| 原始 Qwen | clean | 0.609 [0.484, 0.719] | 0.468 / 0.417 | 0.684 / 0.260 | 0.684 | — | 0.047 | 8 |
| 原始 Qwen | reverb | 0.562 [0.438, 0.688] | 0.435 / 0.375 | 0.643 / 0.240 | 0.661 | — | 0.101 | 9 |
| 原始 Qwen | spatial | 0.500 [0.359, 0.625] | 0.371 / 0.375 | 0.571 / 0.213 | 0.589 | 0.016 | 0.141 | 9 |
| 原始 Qwen | direction | 0.328 [0.203, 0.469] | 0.274 / 0.167 | 0.375 / 0.140 | 0.661 | 0.258 | 0.418 | 8 |
| Audio LoRA | clean | 0.781 [0.687, 0.875] | 0.419 / 0.917 | 0.758 / 0.333 | 0.758 | — | 0.027 | 0 |
| Audio LoRA | reverb | 0.766 [0.656, 0.875] | 0.403 / 0.917 | 0.742 / 0.327 | 0.758 | — | 0.067 | 0 |
| Audio LoRA | spatial | 0.641 [0.500, 0.766] | 0.323 / 0.833 | 0.612 / 0.273 | 0.642 | 0.016 | 0.104 | 0 |
| Audio LoRA | direction | 0.453 [0.312, 0.594] | 0.258 / 0.500 | 0.433 / 0.193 | 0.672 | 0.258 | 0.384 | 0 |
| Whisper+T2-LoRA | clean | 0.734 [0.625, 0.844] | 0.419 / 0.875 | 0.797 / 0.313 | 0.831 | — | 0.057（Whisper 0.088） | 6 |
| Whisper+T2-LoRA | reverb | 0.688 [0.578, 0.797] | 0.387 / 0.833 | 0.721 / 0.293 | 0.770 | — | 0.165（0.165） | 4 |
| Whisper+T2-LoRA | spatial | 0.641 [0.516, 0.766] | 0.355 / 0.792 | 0.683 / 0.273 | 0.750 | 0.016 | 0.108（0.138） | 4 |
| Whisper+T2-LoRA | direction | 0.406 [0.281, 0.547] | 0.242 / 0.458 | 0.433 / 0.173 | 0.750 | 0.274 | 0.394（0.428） | 4 |

- 三条路线解析失败均为 0，拒答均为 0。action 召回：Audio LoRA 0.016–0.031，其余路线 0。
- 方向条件只计作答样本：target-any 原始 Qwen 0.256、Audio LoRA 0.419、T2-LoRA 0.372；只命中干扰者 0.357 / 0.357 / 0.381。
- 描述性配对差（Audio LoRA − Whisper+T2-LoRA，target-any）：clean +0.047 [−0.031, +0.125]，reverb +0.078 [0.000, +0.172]，spatial 0.000 [−0.078, +0.094]，direction +0.047 [−0.016, +0.109]。区间全部包含或触及 0。pilot 每个条件只有 64 条，**只作描述，不作判定**。
