English translation of c0_intent_diag_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c0_intent_diag_v1.md

# Intent differences between C0 and C1/P2 (c0-intent-diag-v1; 2026-11-06; reads existing outputs only, descriptive only)

> Reads two existing outputs only; no inference, no change to T2 or P2, no change to the scoring rules. Description only, no conclusions.

## Inputs and method
- C0: `outputs/qwen_omni_c0/22023448/C0/raw.jsonl` (reference transcripts + T2); C1/P2: `outputs/qwen_omni_c3_p2/22014089/C1/raw.jsonl` (clean recordings + P2). The same 124 recordings; gold intents and sentences checked to agree item by item.
- Whether the intent is correct: strict intent from `score_item` of slu-score-v2 (the same as the intent accuracy in the C0 / C1 reports).
- Script: `scripts/qwen/c0_intent_diag.py` (commit `0dddea4`). srun 22023498 (<CPU_PARTITION>, account <SLURM_ACCOUNT>, 4 CPU, 16 GB), node <NODE>. Results: `outputs/c0_intent_diag_v1/c0_intent_diag.json` (not in the repository).

## Counts
| | Count |
|---|---|
| both correct | 88 |
| **C0 wrong, P2 correct** | **7** |
| **P2 wrong, C0 correct** | **3** |
| both wrong | 26 |
- Intent accuracy C0 0.734 (91 / 124), P2 0.766 (95 / 124), a difference of 4 items, exactly equal to 7 − 3.

## C0 wrong, P2 correct (all 7)
| Recording | Gold intent | C0 prediction | Reference transcript (C0 input) | P2's own transcript |
|---|---|---|---|---|
| audio-1490198212 | qa_factoid | general_query | where is george w bush from | Where is George W. Bush from? |
| audio-1490199179 | email_sendemail | email_query | reply to mark's email | reply to mark's email |
| audio-1490261527 | qa_factoid | datetime_query | what year was bill pullman born | What year was Bill Pullman born? |
| audio-1490366027 | qa_definition | cooking_query | what is a hibachi grill | What is a hibachi grill? |
| audio-1490368934 | iot_wemo_on | general_greet | on mobile charger plug | on off my bell charger plug |
| audio-1497270966 | qa_factoid | general_query | how big is japan | How big is Japan? |
| audio-1499244634 | play_music | music_query | play the latest pop song | Play the latest pop song |

## P2 wrong, C0 correct (all 3)
| Recording | Gold intent | P2 prediction | Reference transcript | P2's own transcript |
|---|---|---|---|---|
| audio-1498572825 | weather_query | general_greet | do i need to bring a coat today | I need to bring a coat today. |
| audio-1499244312 | calendar_query | alarm_query | what are my reminders today | What are my reminders today? |
| audio-1501694638 | qa_stock | general_quirky | what price google up to | What price Google up to |

## Distribution and features (description only)
- **Scenario**: of the 7 "C0 wrong, P2 correct" items, 4 have gold scenario `qa` (qa_factoid 3, qa_definition 1), and email, iot and play have 1 each. On these 4 qa items C0 predicted general_query (2), datetime_query and cooking_query. Only 1 of the 7 (email_sendemail → email_query) has the right scenario. The 3 "P2 wrong, C0 correct" items each have a different scenario (weather, calendar, qa), and P2 got the scenario wrong on all 3.
- **The 26 items both got wrong**: play 12, qa 9, email 2, calendar / takeaway / cooking 1 each.
- **Sentence length**: the 7 "C0 wrong, P2 correct" items average 4.9 words, none ≤ 2 words; the 3 "P2 wrong, C0 correct" items average 6.0 words; all 124 items average 6.0 words, with 4 items of ≤ 2 words. The "indie" item (1 word, C0 parse failure) has the intent wrong for both and is not among these 10.
- **Punctuation and case**: the reference transcripts are all lowercase without punctuation (SLURP original text). Of P2's own transcripts, 109 of the 124 contain punctuation or capitals. Of the 7 "C0 wrong, P2 correct" items above, 4 have capitalised person or place names in the P2 transcript (George W. Bush, Bill Pullman, Japan) or a question mark; this is only co-occurrence, and whether it is a cause was not tested.
- **Other differences**: C0 and P2 differ not only in whether the input is text or audio, but also in the prompt (T2 and P2 differ only in the first sentence, see `qwen_omni_c1_p2t2.py`). In addition, 1 reference transcript itself does not match the recording (audio-1490368934: reference transcript "on mobile charger plug", P2 heard "on off my bell charger plug").
- The difference of 4 items (3.2 percentage points) comes from a very small sample; no significance test was done.

## Record (2026-11-06, decision of the project lead)
- The intent-accuracy difference between C0 and P2 (4 items) is recorded as **"possibly small-sample fluctuation"**; T2 is not changed and it is not investigated further.
