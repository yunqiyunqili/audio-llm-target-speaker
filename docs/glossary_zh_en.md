# 术语表 / Chinese–English glossary

docs/ 下英文译本（`*.en.md`）统一使用下列译法，以中文原文为准。
The English translations (`*.en.md`) under docs/ use the renderings below consistently; the Chinese originals are authoritative.

## 指标与评分 / Metrics and scoring
| 中文 | English |
|---|---|
| 目标至少命中一个 | target-any |
| 只命中目标 | target-only |
| 只命中干扰者 | interferer-only |
| 配对翻转（两次交换都只命中目标） | pair flip (target-only in both swaps) |
| 目标值命中 / 干扰者值命中 | target-value hit / interferer-value hit |
| 可区分的值 | distinguishable values |
| 金标实体值 | gold entity values |
| 意图严格准确率 | strict intent accuracy |
| 实体 F1（严格） | entity F1 (strict) |
| 零实体输出 | zero-entity outputs |
| 主指标 | primary metric |
| 主要比较 | primary comparison |
| 等量（raw）口径 | equal-weight (raw) basis |
| 自然夹角加权 | natural angle weighting |
| 口径 | basis / counting convention |
| 配对 bootstrap；成组重抽样；分层 | paired bootstrap; grouped (cluster) resampling; stratified |
| 区间跨 0 | interval crosses 0 |
| 下界 | lower bound |
| 显著 | significant |
| 只作描述 | descriptive only |
| 事后探索性分析 | post-hoc exploratory analysis |
| 覆盖率 / 风险 | coverage / risk |
| 风险-覆盖率 | risk–coverage |
| 退化阈值 | degradation threshold |
| 门控 | gate |
| 判定 | verdict |
| 规范化 | normalisation |

## 数据与条件 / Data and conditions
| 中文 | English |
|---|---|
| 干净语音 / 干净近讲录音 | clean speech / clean close-talk recording |
| 混响语音；完整混响；早期混响 | reverberant speech; full reverberation; early reverberation |
| 分离语音 | separated speech |
| 单人 / 两人 | single-talker / two-talker |
| 两人混合；两人重叠 | two-talker mixture; two-talker overlap |
| 未分离混合 | unseparated mixture |
| 近讲 | close-talk |
| 目标说话人 / 干扰者 | target speaker / interferer |
| 交换目标 | target swap |
| 夹角层 | angle stratum (plural: strata) |
| 询问方向 / 询问方位 | query direction |
| 方向说明 | direction instruction |
| 扇区 | sector |
| 选择半 / 报告半 | selection half / report half |
| 划分 | split |
| 条（录音）/ 句（文本）/ 组 / 个样本 | recordings / sentences / groups (pairs) / samples |
| 冻结；冻结点；冻结文档 | frozen; freeze point; frozen document |
| 运行前写定，看到结果后不修改 | written before the run and not changed after the results were seen |
| 审批 / 批准 | approval / approved |
| 项目负责人 | project lead |
| 作业号 | job ID |

## 系统 / System
| 中文 | English |
|---|---|
| 前端；分离前端 | front end; separation front end |
| 选路 | stream selection |
| 路（分离输出的一路） | stream |
| 选中路 / 目标路 | selected stream / target stream |
| 路由正确 / 错误 | correctly / incorrectly routed |
| 极性修正 | polarity correction |
| 方向估计；方向匹配 | direction estimation; direction matching |
| 参考匹配选路 | reference-matched stream selection |
| 拒答 | abstention |
| 拒答 A（margin）/ 拒答 B（扇区） | abstention A (margin) / abstention B (sector) |
| 方向不可辨 | direction indistinguishable |
| 音频路线 / 级联路线 | audio route / cascaded route |
| 原始 Qwen / 原始 P2 | original Qwen / original P2 |
| 音频 LoRA / 文本 LoRA（T2-LoRA） | audio LoRA / text LoRA (T2-LoRA) |
| 只对回答部分计损失 | loss on the answer tokens only |
| 标准转写 | reference transcript |
| 映射表；意图默认对象；意图 → 动作 | mapping table; intent-default object; intent → action |
| 快照 | snapshot |
| 先导测试 | pilot test |
| 可靠性选流 | reliability-based stream selection |
| LLM 级融合 | LLM-level fusion |

## 保留不译 / Kept as is
条件名（C0–C7、C1-R、C4-direct、C6δ、R1、R1-same、F1）、方法名（M1/M2/M3）、prompt 名（P2、T2）、规则与版本名（binding-score-v1、slu-score-v2、mix_v1、c6-real-v1 等）、数据集与模型名、代码、路径、命令、文件名、sha256、提交号、作业号。
Condition names (C0–C7, C1-R, C4-direct, C6δ, R1, R1-same, F1), method names (M1/M2/M3), prompt names (P2, T2), rule and version names (binding-score-v1, slu-score-v2, mix_v1, c6-real-v1, etc.), dataset and model names, code, paths, commands, file names, sha256 values, commit hashes and job IDs are kept unchanged.
