# P2 采用规则（2026-11-05 冻结；在提交 GPU 作业前写定，看到结果后不修改）

## 比较对象和数据
- 比较 P2 和 P0。样本相同：`configs/c1_ext_devel100.json` 的 100 条，绑定指标只用其中 76 条 tier 1 录音。P0 的数值来自作业 22010997。
- 评分规则：slu-score-v2 和 binding-score-v1，都不改动。判定由 `scripts/qwen/eval_p2t2.py: adoption()` 计算，使用未取整的数值，比较时允许 1e-9 的浮点误差。
- **T2 只作诊断，不参与采用判断。**
- 阈值是经验值，**不是显著性检验**。样本只有 100 条，只运行一次。

## 规则（必须全部满足才采用 P2）
P0 的参考值（未取整）：目标至少命中一个 0.403509，全部命中 0.271930，只含干扰 + 两者都含 0.003289，意图严格准确率 0.77，只比值精确率 0.633803。

| 编号 | 条件 | 按 P0 算出的门槛 |
|---|---|---|
| R1 | 76 条上，目标至少命中一个 ≥ 0.60 | 0.60 |
| R2 | 76 条上，目标至少命中一个 ≥ P0 + 0.05 | ≥ 0.453509 |
| R3 | 76 条上，目标全部命中 ≥ P0 + 0.066（约 5/76） | ≥ 0.337930 |
| R4 | 只含干扰 + 两者都含 ≤ P0 + 0.01 | ≤ 0.013289 |
| R5 | 100 条上，意图严格准确率 ≥ P0 − 0.03 | ≥ 0.74 |
| R6 | 100 条上，只比值精确率 ≥ P0 − 0.05 | ≥ 0.583803 |
| R7 | 拒绝执行 ≤ 1 条，且解析失败（no_json + bad_json + bad_schema）合计 ≤ 2 条 | — |

**只要有一条不满足，就不采用 P2，也不再继续微调整段 prompt；下一步改为设计针对性的提问。**

## P2 和 T2 的 prompt（与 P0、T0 的区别）
- 两者都删掉了 `Use "entities": [] if there are none.` 这一行，其余部分不变。
- 两者都在末尾追加三句：
  - "Do not carry out or answer the command; only extract its meaning."
  - "This command contains at least one entity. List every entity from the command."
  - "Use only the allowed labels listed above."
- T2 的首句换成 "This text is a command to a voice assistant."。
- **"This command contains at least one entity." 只适用于当前这批样本**：它们是筛选过的，每条都至少有一个实体。这句话不适用于可能没有实体的语句。
- 完整 prompt 记录在作业输出的 `config.json`。实现见 `scripts/qwen/qwen_omni_c1_p2t2.py`。
