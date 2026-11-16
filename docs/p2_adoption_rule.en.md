English translation of p2_adoption_rule.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：p2_adoption_rule.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# P2 adoption rule (frozen 2026-11-05; written before submitting the GPU job and not changed after the results were seen)

## Comparison and data
- P2 is compared with P0. Same samples: the 100 items of `configs/c1_ext_devel100.json`; the binding metrics use only the 76 tier 1 recordings among them. The P0 values come from job 22010997.
- Scoring rules: slu-score-v2 and binding-score-v1, both unchanged. The verdict is computed by `scripts/qwen/eval_p2t2.py: adoption()`, using unrounded values and allowing a floating-point tolerance of 1e-9 in comparisons.
- **T2 is a diagnostic only and takes no part in the adoption decision.**
- The thresholds are empirical, **not significance tests**. There are only 100 samples, run only once.

## Rules (all must be satisfied for P2 to be adopted)
Reference values of P0 (unrounded): target-any 0.403509, all target values hit 0.271930, interferer-only + both 0.003289, strict intent accuracy 0.77, value-only precision 0.633803.

| ID | Condition | Threshold computed from P0 |
|---|---|---|
| R1 | on the 76, target-any ≥ 0.60 | 0.60 |
| R2 | on the 76, target-any ≥ P0 + 0.05 | ≥ 0.453509 |
| R3 | on the 76, all target values hit ≥ P0 + 0.066 (about 5/76) | ≥ 0.337930 |
| R4 | interferer-only + both ≤ P0 + 0.01 | ≤ 0.013289 |
| R5 | on the 100, strict intent accuracy ≥ P0 − 0.03 | ≥ 0.74 |
| R6 | on the 100, value-only precision ≥ P0 − 0.05 | ≥ 0.583803 |
| R7 | refusals ≤ 1, and parse failures (no_json + bad_json + bad_schema) ≤ 2 in total | — |

**If any one rule is not satisfied, P2 is not adopted, and no further fine-tuning of the whole prompt is done; the next step switches to designing targeted questions.**

## The P2 and T2 prompts (differences from P0, T0)
- Both delete the line `Use "entities": [] if there are none.`; everything else is unchanged.
- Both append three sentences at the end:
  - "Do not carry out or answer the command; only extract its meaning."
  - "This command contains at least one entity. List every entity from the command."
  - "Use only the allowed labels listed above."
- The first sentence of T2 is replaced by "This text is a command to a voice assistant.".
- **"This command contains at least one entity." applies only to the current batch of samples**: they were screened, and each has at least one entity. The sentence does not apply to utterances that may have no entities.
- The full prompts are recorded in `config.json` of the job outputs. Implementation in `scripts/qwen/qwen_omni_c1_p2t2.py`.
