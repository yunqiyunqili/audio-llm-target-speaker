English translation of c0_date_halves_diag_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c0_date_halves_diag_v1.md

# Difference in date recall between the two halves of C0-428 (c0-date-halves-diag-v1; 2026-11-06; read-only, descriptive only, split not changed)

- Input: C0-428 raw output `outputs/qwen_omni_c0_devel428/22023517/C0/raw.jsonl`; split per `docs/lora_slu_plan_v2.md` §3.2 (the 382 devel slurp_ids shuffled with `random.Random(2026)`, the first 191 forming the selection half). Classification uses `eval_c0_devel428.classify` (c1-error-taxonomy-v1 rules, unchanged).
- Script `scripts/qwen/c0_date_halves_diag.py` (commit `c8adc2a`); srun 22023770 (<CPU_PARTITION>, account <SLURM_ACCOUNT>, 4 CPU, 16 GB), node <NODE>. Results: `outputs/c0_date_halves_diag_v1/` (not in the repository).

## Counts
| | Selection half | Report half | All |
|---|---|---|---|
| date gold entities | 63 | 65 | 128 |
| sentences / recordings containing date | 53 / 62 | 50 / 64 | 103 / 126 |
| recall (per recording) | **0.365** (23 / 63) | **0.631** (41 / 65) | 0.500 |
| recall (each sentence counted once) | 0.407 (54) | 0.627 (51) | 0.514 |
| miss classes: c boundary / d type / e omission | **11** / 6 / 23 | **1** / 4 / 19 | 12 / 10 / 42 |
| share of single-word dates; recall of single-word / multi-word | 0.65; 0.41 / 0.27 | 0.62; 0.58 / 0.72 | 0.63; 0.49 / 0.51 |
- The gap remains after removing repeated recordings (0.407 vs 0.627), so it is not caused by a few repeated sentences.

## Most common date values (count / hits)
- Selection half: today 15 / 6, tomorrow 7 / 3, **today 's 7 / 0**, this week 5 / 2, wednesday 4 / 3, **tuesday 3 / 0**, next week 3 / 3, **fifteenth 3 / 0**, thursday 3 / 1, february 2 / 2.
- Report half: today 25 / 12, tomorrow 9 / 5, this week 6 / 5, next week 4 / 4, friday 3 / 3, today 's 2 / 0, twenty second 2 / 2, this weekend 2 / 0.

## By intent (count / hits)
- Selection half: weather_query 12 / 3, **datetime_query 12 / 4**, calendar_set 11 / 4, calendar_query 11 / 9, calendar_remove 5 / 1, news_query 3 / 0.
- Report half: **calendar_query 17 / 16**, weather_query 15 / 5, **calendar_set 8 / 8**, general_quirky 6 / 1, datetime_query 5 / 4, news_query 5 / 4.

## Examples of omissions (the C0 input is the reference transcript)
- Selection half:
  - e: `set a birthday event for wednesday` (only event_name extracted); `will it be cold tomorrow`; `give me the information of today's cricket match`; `what was the weather report for today`; `will it rain at all this week`; `what were the winning powerball numbers last night`; `i want to hear yesterday's podcast from my favorite channel`.
  - c (boundary): `... next tuesday at ten pm` (predicted `next tuesday`, gold `tuesday`); `... for the fifteenth` (predicted `the fifteenth`); `what is today's date` (predicted `today's date`, gold `today 's`).
  - d (type): `... at one pm today` (predicted `time: one pm today`); `... on thursday afternoon` (predicted `datetime: thursday afternoon`).
- Report half (almost all e): `is it going to rain tomorrow`, `what is the weather today`, `is the meeting scheduled for tomorrow`, `did i get any emails from sam today`, `i went skiing today`, `will i need sunblock this weekend`.

## Possible reasons (description only, not tested)
1. **Different intent composition**: dates in the report half are concentrated in calendar_query and calendar_set (24 of 25 hit), intents for which the model almost always extracts the date; dates in the selection half appear more in datetime_query, weather_query and news_query (7 of 27 hit), intents in which the date is often omitted or merged into another entity.
2. **Boundary mismatches are concentrated in the selection half**: 11 vs 1 for class c, mainly because the SLURP gold value is shorter than the model output (`tuesday` vs `next tuesday`, `fifteenth` vs `the fifteenth`, `today 's` vs `today's date`). The selection half has 7 instances of `today 's`, all missed; the report half has only 2.
3. **Small sample size**: each half has only about 50 sentences containing date, so a single sentence template (e.g. several recordings of "what is today's date") can shift the result by several percentage points.
- The above only describes the difference between the halves; the split is unchanged. When selecting on the selection half, the date-recall baseline is lower than on the report half, and the date recall of the two halves should not be compared directly.
