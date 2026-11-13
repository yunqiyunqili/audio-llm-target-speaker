# 数据与模型许可声明

本仓库**不包含任何音频、模型权重或 checkpoint**。代码与文档的许可见 `LICENSE`（MIT）；MIT 许可不覆盖下列数据。

## SLURP
- 来源：Bastianelli, E., Vanzo, A., Swietojanski, P., Rieser, V. "SLURP: A Spoken Language Understanding Resource Package." EMNLP 2020.
- **文本标注：CC BY 4.0**。本仓库的 `manifests/slurp_split_v2_seed2026/`、`configs/`（如 `mix_v1_*.json`、`c1_*.json`）和部分文档含 SLURP 句子、意图和实体，按 CC BY 4.0 使用，并在此署名。
- **音频：CC BY-NC 4.0**。本仓库**不分发**任何 SLURP 音频，也不分发由其生成的衍生音频（混响、混合、分离输出）。`results/slurp/headset_durations.csv` 只含录音文件名和时长。

## SMS-WSJ / WSJ
- 来源：Drude, L., Heitkaemper, J., Boeddeker, C., Haeb-Umbach, R. "SMS-WSJ: Database, performance measures, and baseline recipe for multi-channel source separation and recognition." arXiv:1910.13934, 2019。
- 本项目只使用 SMS-WSJ 的模拟房间脉冲响应（RIR）和几何元数据。仓库中只有 RIR 样例 ID 和由几何计算的统计（`results/sms_wsj_geometry/`），**不含 RIR 文件**。
- 底层 WSJ0 / WSJ1 语音由 LDC 授权，本项目没有使用其语音，也不分发。SMS-WSJ 工具的许可以其官方仓库为准。

## Fluent Speech Commands（FSC）
- 来源：Lugosch, L., Ravanelli, M., Ignoto, P., Tomar, V. S., Bengio, Y. "Speech Model Pre-training for End-to-End Spoken Language Understanding." Interspeech 2019。
- **获取**：请从 Fluent.ai 官方渠道获取 FSC（官方数据页、入口和许可见 `configs/fsc_pilot_pending.json` 的 `official_*` 字段）。本仓库不提供任何第三方下载地址。
- 本项目只将 FSC 用于**非商业学术研究**；所用副本的再分发授权**尚未得到 Fluent.ai 确认**，发表前需确认。
- 本仓库只给**汇总数字**和标签级的映射规则（含 FSC train 的汇总词频），**不含任何逐条路径、说话人 ID、原句或逐条模型输出**；不分发任何 FSC 音频。FSC official test 没有使用。

## 模型
- Qwen2.5-Omni-7B、Whisper large-v3、SpatialNet：许可以各自的模型卡或官方仓库为准。本仓库不含任何权重；LoRA 适配器也不在仓库中。
