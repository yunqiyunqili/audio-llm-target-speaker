English translation of scoring_rules.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：scoring_rules.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# SLU scoring rules (slu-score-v2, frozen 2026-11-05)

**Freeze convention**: this set of rules is frozen before entering test. Any later change must bump the version number (v3), recompute all devel results with the same new rules, and state clearly what changed. No changes after test has been used. Result files of old versions are never overwritten.

## Frozen files
| File | sha256 |
|---|---|
| `scripts/qwen/slu_normalize.py` (parsing and v1 normalisation) | `d623281c7bd83029197dfca3bc99927b5cac951f12562309d343a77247ed72c3` |
| `scripts/qwen/rescore_slu_v2.py` (v2 scoring) | `00ad15d80a9688825988767adcbe67f246609cc0fc061a6be3f259ba0e8fa498` |
| `scripts/qwen/compare_c1_ext_conditions.py` (item-by-item comparison between conditions, no scoring) | `a77b20eaffe42fc6795a959292b2c682a09e468cb7bd00564cf7cc4a76dd98c3` |
| `configs/slurp_label_inventory_train.json` (60 intents, 55 entity types, from the official train) | unchanged |

## Input
Only `raw.jsonl` (raw model output) is read; nothing is regenerated. Reference labels come from SLURP commit 8eb1654; the intent is `scenario_action`, entities are (type, value), with value formed by joining the tokens of the SLURP annotation.

## Parsing (same as v1)
Remove the ``` code-block markers, take the content between the first `{` and the last `}`, and parse it with `json.loads`. Status is one of ok / no_json / bad_json / bad_schema; when not ok, the intent counts as wrong and the entities as empty. type is lowercased; entities whose type or value is not a string are discarded and counted.

## Normalisation
- v1 (`normalize`): see `docs/c1_semantic_plan.md`.
- v2 (`normalize_v2`): first apply v1, then unify `rnb`, `r'n'b`, `r and b`, `r n b` as `r n b`.
- Possessives: when judging entity boundaries and computing word overlap, a word-final `'s` is removed before comparison ("derek's" and "derek" share a word). Strict matching does **not** remove it.

## Metrics
| Metric | Definition |
|---|---|
| Strict intent accuracy | parsed ok, and the intent is exactly identical to scenario_action |
| Scenario-prefix accuracy | the part of the intent before the first `_` is the same |
| Strict entity F1 (v2) | (type, normalize_v2(value)) matched as multisets; `strict_v1` uses v1 normalisation, kept for comparison with old results |
| Value-only F1 | type ignored, only normalize_v2(value) matched as multisets. **No** manual mapping of similar types |
| Word-overlap F1 | same type and (after removing possessives) at least one shared word; greedy one-to-one matching in reference order |
| Strict full-semantics accuracy | intent strictly correct **and** entity multiset exactly identical to the reference (v2 normalisation); transcript not considered. The value under v1 normalisation is also reported |
| Transcript WER | word edit distance after normalize_v2; `wer_v1` uses v1 normalisation. Reported only for audio conditions |
| Other counts | parse status, discarded entities, entity types and intents outside the inventory, number of items reaching max_new_tokens |

## Automatic error types (several per item possible)
`parse`; `transcript` (WER>0, audio conditions only); `intent`; for entities, strict matches are removed first, and the remaining reference entities are classified in turn as: `entity_type` (same value, different type) → `entity_boundary` (same type, shared word, including possessive differences) → `entity_missing`; remaining output entities that neither share a value with nor are of the same type with a shared word as the remaining reference entities are classed as `entity_spurious`. "Annotation problems" are not judged automatically.

## Disputed items
Samples listed in `configs/c1_disputed_items.json` are marked `review_status = 待确认` in the error details, **with scores unchanged**. Per the decision of 2026-11-05, no audio is listened to; spot checks are made after the 100-item results come out.

## Validation (2026-11-05, the 20 items of job 22010083)
- `strict_v1` agrees with the v1 `rescore_summary.json`: TP 10, F1 0.385; the md5 of the v1 file is unchanged before and after rescoring.
- v2: strict entity F1 0.423 (TP 11; the extra 1 is "RnB"), value-only 0.615, word overlap 0.538, strict full semantics 4/20 (3/20 under v1 normalisation), WER 0.032 (0.052 under v1 normalisation).
