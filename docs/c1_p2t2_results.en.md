English translation of c1_p2t2_results.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c1_p2t2_results.md

# P2 / T2 results and prompt freeze (job 22011390; 2026-11-05)

## Scope (important)
- These are results of **a single run** on **100 devel items** (`configs/c1_ext_devel100.json`) and do not represent overall performance.
- P2 was chosen after comparison with P0 on **the same batch of samples**, so the results **may be optimistic**.
- The sentence "This command contains at least one entity." in the prompt applies only to **samples with entities**. These 100 items were screened, and each has entities. The sentence does not apply to utterances without entities.

## Freeze decision (2026-11-05)
- Subsequent **audio conditions use the P2 prompt** and **text conditions use the T2 prompt** (full text in `outputs/qwen_omni_c1_p2t2/22011390/config.json`; implementation in `PROMPT_P2` and `PROMPT_T2` of `scripts/qwen/qwen_omni_c1_p2t2.py`).
- Entity types outside the inventory are **only counted separately; the prompt is not changed**.
- The adoption rule is in `docs/p2_adoption_rule.md`. The rule was frozen before job submission, and all 7 items pass.

## Run record
- Job 22011390, node <NODE>, run time 2026-11-05 10:31:30–10:37:29 (5 min 59 s), exit code 0, peak process memory (MaxRSS) 26.7 GB.
- Loading 50.7 s. Inference time P2 117.9 s, T2 112.9 s; inference GPU memory peak 16.94 / 16.85 GiB.
- Code version: commit `23e85d2`, no uncommitted dependency files at run time. The sha256 of `qwen_omni_c1_p2t2.py` is `9f35c58b…`, with the full value in config.json.
- Evaluation: `scripts/qwen/eval_p2t2.py`, results in `outputs/qwen_omni_c1_p2t2_eval/22011390/eval_summary.json`; transcript diagnostic: `scripts/qwen/p2_transcript_diag.py`, results in `outputs/qwen_omni_c1_p2t2_eval/22011390_transcript_diag/`. Both only read existing raw outputs and were run on the login node.

## Binding metrics (76 tier 1 recordings; parsed entities only; computed per recording, then averaged over recordings)
| | Target only | Interferer only | Both | Neither | target-any | All target values hit |
|---|---|---|---|---|---|---|
| P0 | 0.272 | 0.003 | 0 | 0.725 | 0.404 | 0.272 |
| P2 | 0.478 | 0.003 | 0 | 0.519 | 0.640 | 0.478 |
| T0 | 0.298 | 0.003 | 0 | 0.698 | 0.417 | 0.298 |
| T2 | 0.588 | 0 | 0 | 0.412 | 0.706 | 0.588 |

## Metrics on the 100 items
| | Strict intent accuracy | Value-only P / R / F1 | Zero entities | Refusals | Parse failures | Out-of-inventory intents / entity types |
|---|---|---|---|---|---|---|
| P0 | 0.77 | 0.634 / 0.352 / 0.452 | 32 | 0 | 0 | 3 / 4 |
| P2 | 0.76 | 0.670 / 0.539 / 0.597 | 4 | 0 | 0 | 2 / 9 |
| T0 | 0.71 | 0.726 / 0.414 / 0.527 | 34 | 1 | 1 | 5 / 5 |
| T2 | 0.75 | 0.736 / 0.609 / 0.667 | 4 | 0 | 0 | 3 / 6 |

## Adoption-rule verdict (comparing P2 with P0 only)
| Rule | Value of P2 | Threshold | Result |
|---|---|---|---|
| R1 target-any ≥ 0.60 | 0.6404 | 0.60 | pass |
| R2 target-any ≥ P0 + 0.05 | 0.6404 | 0.4535 | pass |
| R3 all target values hit ≥ P0 + 0.066 | 0.4781 | 0.3379 | pass |
| R4 interferer-only + both ≤ P0 + 0.01 | 0.0033 | 0.0133 | pass |
| R5 strict intent accuracy ≥ P0 − 0.03 | 0.76 | 0.74 | pass |
| R6 value-only precision ≥ P0 − 0.05 | 0.6699 | 0.5838 | pass |
| R7 refusals ≤ 1, parse failures ≤ 2 | 0 / 0 | 1 / 2 | pass |

## Transcript diagnostic: P2 vs P0 (existing raw outputs only)
| | Transcript WER (v2 normalisation) | Items with transcript errors | target-any in transcript | Target only in transcript |
|---|---|---|---|---|
| P0 | 0.0579 (36/622) | 26 | 0.921 | 0.851 |
| P2 | 0.0595 (37/622) | 26 | 0.917 | 0.842 |

The transcript diagnostic is for reference only and is not a primary metric. P2 and P0 are essentially the same on transcription; the improvement in the binding metrics comes from more entities being output (zero-entity outputs drop from 32 to 4).
