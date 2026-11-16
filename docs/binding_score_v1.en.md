English translation of binding_score_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：binding_score_v1.md

# Key entity value binding score v1 (binding-score-v1, 2026-11-05)

The definition of the primary metric is in `docs/conditions.md`. Implementation: `scripts/slurp/binding_score_v1.py`. Normalisation uses normalize_v2 from `docs/scoring_rules.md`.

## Definitions
- Pair (T, I): v2 devel candidate pairs (same split, same intent, same multiset of entity types, different entity values, different speakers), also of the same gender with duration difference ≤ 1 s.
- D_T = T − I, D_I = I − T: multiset subtraction of entity values after normalize_v2. Types are not compared, because the two utterances have the same type multiset. A pair is valid only if D_T and D_I are **both non-empty**.
- Primary metric: **only the values of the parsed entities are considered**, matched as complete values (multisets).
  - target hit = all values in D_T appear (counting repetitions); interferer hit = any value in D_I appears.
  - Four classes: target only (target hit and no interferer hit), interferer only, both, neither.
  - Also reported: target-any; target hit rate = |D_T ∩ H| / |D_T|.
- Diagnostic (not the primary metric; reported only for P0, P1): counts "the value appears as a complete word string in the normalised transcript" as a hit, with the other rules the same.
- On parse failure, the output entities are taken as the empty set.
- Aggregation: first, for each target recording, compute the class proportions and hit rate over all its valid interferers, then take the unweighted mean over recordings. "Range" means the minimum and maximum of each recording's hit rate across different interferers.

## Scope
Currently there are only results for C1 clean speech (job 22010997), which only check which values appear in the target speaker's own clean output. **These are not mixture results**, and pair flips are not computed.

## Results (22010997; targets are the 76 tier 1 recordings in configs/c1_ext_devel100.json)
- All 76 recordings have at least 1 valid interferer; 190 (target, interferer) combinations in total, 1–8 interferers per recording.
- Output: `outputs/qwen_omni_c1_ext_post/22010997/binding_v1/`.

| Condition / method | Target only | Interferer only | Both | Neither | target-any | Mean hit rate (mean of per-recording min / mean of max) | Recordings whose hit rate varies with the interferer |
|---|---|---|---|---|---|---|---|
| P0 entities | 0.272 | 0.003 | 0 | 0.725 | 0.404 | 0.338 (0.329 / 0.349) | 2 |
| P1 entities | 0.237 | 0 | 0 | 0.763 | 0.368 | 0.303 (0.303 / 0.303) | 0 |
| T0 entities | 0.298 | 0.003 | 0 | 0.698 | 0.417 | 0.357 (0.349 / 0.368) | 2 |
| P0 transcript (diagnostic) | 0.851 | 0 | 0.004 | 0.145 | 0.921 | 0.888 | 0 |
| P1 transcript (diagnostic) | 0.816 | 0.004 | 0 | 0.180 | 0.890 | 0.853 (0.849 / 0.855) | 1 |

- Every individual recording's hit rate lies between 0 and 1.
- The few cases of "interferer hit" are all artefacts of complete-value matching itself, not real binding to the interferer:
  - P0 and T0: the reference value is "tomorrows", the model outputs "tomorrow", which happens to equal one interferer's value.
  - Transcript diagnostic: the interferer's value "john" appears as a word in "tara john".
