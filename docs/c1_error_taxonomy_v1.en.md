English translation of c1_error_taxonomy_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c1_error_taxonomy_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C1 error taxonomy v1 (c1-error-taxonomy-v1; 2026-11-06; rules written before the run and not changed after the results were seen)

> - Purpose: determine whether P2's failure to hit target entities on C1 (clean close-talk recordings) is mainly "different spelling" or "really not extracted correctly".
> - **Reads existing outputs only, runs on CPU.** No inference, no training, no downloads, no environment changes, training jobs not touched, test not touched. mix_v1, the P2/T2 prompts, slu-score-v2, binding-score-v1, the v3 stream-selection rule, existing results and verdicts are not changed.
> - The normalisation in this document ("diagnostic normalisation") is **used only for this diagnosis** and does not replace normalize_v2 or binding-score-v1.

## 1. Inputs
| Route | File | Source |
|---|---|---|
| C1 / P2 | `outputs/qwen_omni_c3_p2/22014089/C1/raw.jsonl` | job 22014089 (`docs/c3_results_v1.md`, "C1 / P2 0.694"; the value is stored in `C1.mean_target_any` of `outputs/qwen_omni_c3_p2_eval/22014089/eval_c3_summary.json`) |
| C7W-C1: T2 output | `outputs/qwen_omni_c7w_f1/22016221/T_C1/raw.jsonl` | job 22016221 (`docs/c7_whisper_v1.md`) |
| C7W-C1: Whisper transcript | `outputs/qwen_omni_c7w_f1/22016221/W_C1/raw.jsonl` | same as above |
- Gold entities: the `entities` of the A and B recordings of each group in `configs/mix_v1_devel100_v2.json` (i.e. the SLURP gold labels used by binding-score-v1).
- Checks: all three files must exist and each cover exactly the 124 recordings of mix_v1 (one line each); the `asr_text` of `T_C1` must agree with the `raw_output` of `W_C1`. **If any item fails, stop and report; no substitute data.**

## 2. Unit and "hit"
- Unit: one entry per gold entity (type, value) of each recording.
- **Hit** (the same matching as binding-score-v1, but not restricted to distinguishing values): the multiset H of predicted entity values after normalize_v2 (types ignored). Check the gold entities in order: if normalize_v2(gold value) still has remaining count in H, it is a hit and consumes one count.
- Only **missed** gold entities are classified. The denominators are also reported: total gold entities, recordings with entities, total recordings; and the classification for the subset "entities that are a target distinguishing value (D_T) in at least one pair" is reported separately.

## 3. Diagnostic normalisation N(s)
In order:
1. lowercase; `a.m.`/`a. m.` → `am`, `p.m.` → `pm`;
2. times: `H:00` → `H`; `H:MM` → `H MM` (H is 1–2 digits); `o'clock` / `oclock` deleted;
3. punctuation (characters other than letters, digits and spaces) replaced by spaces, spaces collapsed, leading and trailing spaces removed;
4. English number words → Arabic numerals: zero–nineteen, twenty–ninety, and "tens + units" combinations (e.g. `twenty five` → `25`); other words unchanged. Ordinals are not converted.
- **Same value**: N(a) = N(b); or **only one** of the two contains the word `am`/`pm` and **the other is written in clock notation in the original** (contains `H:MM`), and they are the same after deleting `am`/`pm`. Thus `7:00`, `7 am` and `seven am` are mutually the same; `7 am` and `7 pm` differ; `seven` and `seven am` do not count as the same (classed c, different boundary, consistent with the example in the task description).
- (Revision note: this item was revised once before the run, after the first commit `9cc3ef9`; the original wording would have judged `seven` and `seven am` the same, contradicting the example in the task description; the run took place only after the revision.)
- **Word overlap**: the word sets of N(a) and N(b) (before deleting `am`/`pm`) intersect. No stop words are set; the number of class c and d cases in which "the overlapping words are all stop words" (the, a, an, of, to, in, on, at, for, my, me, and, is, it, this, that) is reported separately, descriptive only.

## 4. Classification (in order; each missed entity goes into one class only)
| Class | Condition |
|---|---|
| a. parse failure | the `parse()` status of that output is not `ok` |
| b. different spelling | the prediction has an entity of the **same type** with the "same value" as the gold value |
| c. different boundary | the prediction has an entity of the **same type** with "word overlap" but not the "same value" as the gold value |
| d. wrong type | the prediction has an entity of a **different type** with the "same value" or "word overlap" as the gold value |
| e. omission | none of the above, but N(gold value) appears as a contiguous word sequence in N(transcript) |
| f. misheard | none of the above, and N(gold value) is not in N(transcript) |
- Type comparison: equal after lowercasing and stripping leading and trailing spaces.
- Each gold entity is compared independently with all predicted entities (no one-to-one assignment).
- transcript: for P2, the `transcript` in the model's own output JSON; for Whisper + T2, the **Whisper transcript** (`raw_output` of `W_C1`).

## 5. Output
- Entity counts and proportions per class (of missed entities), P2 and Whisper + T2 side by side; one table for all gold entities and one for the D_T subset.
- At most 10 examples per class (the first 10 sorted by recording file name): recording id, gold entity, predicted entities, transcript.
- **target-any with class b counted as hits** (diagnostic only, not replacing binding-score-v1): add the normalize_v2 values of class-b gold values to H, with the rest of the computation unchanged:
  - averaged per recording (the same as `eval_c3_v1.py`: each recording computed over the pairs of the groups it belongs to, then averaged over recordings);
  - mapped to the 200 samples (the same as `eval_c7w_f1.py`: output of the target recording + the sample's own pair, `eval_c4_v1.score(single=True)`).
  - Consistency: without class b, the per-recording value for P2 must equal 0.6935… (`eval_c3_summary.json`), and the value for Whisper + T2 mapped to the 200 samples must equal 0.650 (`C7W_C1` in `outputs/c7w_f1_eval/22016732_full/eval_c7w_f1.json`); otherwise stop.
- No verdict thresholds; descriptive only.

## 6. Implementation and run
- Script: `scripts/qwen/c1_error_taxonomy.py`; results: `outputs/c1_error_taxonomy_v1/` (not in the repository).
- Run with srun on <CPU_PARTITION> with a clean working tree: account <SLURM_ACCOUNT>, 4 CPU, 16 GB, 30 minutes; environment `envs/qwen_omni` (standard library and existing scripts only).

---

## Results (2026-11-06)
- Run record: srun 22023424 (<CPU_PARTITION>, account <SLURM_ACCOUNT>, 4 CPU, 16 GB, 30 minutes), node <NODE>, 13:32:46–13:32:49, exit code 0. At run time HEAD was `88c0351`, working tree clean. Results in `outputs/c1_error_taxonomy_v1/` (`c1_error_taxonomy.json`, `entities_P2.jsonl`, `entities_T2.jsonl`, not in the repository).
- Input checks pass: each of the three files has 124 recordings, consistent with mix_v1; the `asr_text` of T_C1 is identical item by item to W_C1; the `ref_entities` in raw are identical item by item to the gold labels in the mix_v1 configuration.
- Consistency passes: without class b, P2's per-recording target-any = 0.6935 (the same as `eval_c3_summary.json`), and Whisper + T2 mapped to the 200 samples = 0.650 (the same as `eval_c7w_f1.json`).

### Denominators
- 124 recordings, **all** with gold entities; **144** gold entities in total.
- **All** 144 entities are a target distinguishing value (D_T) in at least one pair, so the results for the D_T subset are exactly the same as for all entities; only one table is listed below.
- Hits: P2 86 / 144 (58 missed), Whisper + T2 81 / 144 (63 missed).

### Classification of missed entities
| Class | P2: count (share of missed) | Whisper + T2: count (share of missed) |
|---|---|---|
| a. parse failure | 0 (0%) | 0 (0%) |
| b. different spelling | **0 (0%)** | **0 (0%)** |
| c. different boundary | 11 (19.0%) | 12 (19.0%) |
| d. wrong type | 6 (10.3%) | 6 (9.5%) |
| e. omission (value in the transcript) | **28 (48.3%)** | **30 (47.6%)** |
| f. misheard (value not in the transcript) | 13 (22.4%) | 15 (23.8%) |
| b + c + d | 17 (29.3%) | 18 (28.6%) |
- Class c and d cases where "the overlapping words are all stop words": P2 0, Whisper + T2 1.
- By gold type: of P2's 58 misses, 19 are `date` (43 `date` gold entities in total), of which 17 are class e; of Whisper + T2's 63 misses, 20 are `date`, of which 16 are class e.

### target-any with class b counted as hits (diagnostic only)
| Route | binding-score-v1: per recording / 200 samples | class b counted as hits: per recording / 200 samples |
|---|---|---|
| P2 | 0.694 / 0.695 | 0.694 / 0.695 (class b is 0, unchanged) |
| Whisper + T2 | 0.653 / 0.650 | 0.653 / 0.650 (unchanged) |

### Examples (P2; at most 10 per class, full list in the json)
- **c different boundary**: `person "mark 's"` → predicted `mark`; `person "barack obaba"` → `Barack Obama`; `person "micheal jackson"` → `Michael Jackson`; `game_name "temple runner"` → `Temple Run`; `list_name "school supplies"` → `school supplies list`; `music_genre "emo music"` → `emo`; `device_type "mobile charger plug"` → `bell charger plug`; `weather_descriptor "temperature"` → `highest temperature`; `weather_descriptor "sunscreen"` → `sunscreen day`; `radio_name "next howard stern show"` → `Howard Stern show`.
- **d wrong type**: `house_place "bathroom"` → `device_type "upstairs bathroom light"`; `definition_word "volkswagen golf car"` → `car_model "Volkswagen Golf"`; `podcast_descriptor "stopped last night"` → `podcast_name "that podcast I stopped last night"`; `business_type "dress shows"` → `clothing_type "dress shorts"`; `place_name "within one mile"` → `distance "one mile"`; `podcast_descriptor "next podcast episode"` → `podcast_name "next podcast"`.
- **e omission**: mostly dates not extracted, e.g. "What are my reminders today?" (gold `date today`, predicted `none none`), "Will it be cold tomorrow?" (gold `date tomorrow`, only `weather_descriptor cold` predicted), "Is it going to rain tomorrow?", "Will I need sunblock this weekend?", "What is the weather today?"; also `event_name holiday`, `definition_word "this sunset"`.
- **f misheard**: `artist_name "bob dylan"` (transcript "Play newest pop didn't playlist"), `person "angelina jolie"` ("How old is engineer Joey?"), `music_genre indie` ("India"), `business_name chipotle` ("Close this trip to."), `definition_word regal` ("rego").

### Observations (description only; the rules are not changed because of them)
1. **Class b (different spelling) is 0**: under the diagnostic normalisation, not a single missed entity differs only in the spelling of numbers, times or case. So by these rules, P2's misses on C1 are **not** caused by "different spelling".
2. The largest class is **e omission (about 48%)**: the value is already in the model's own transcript but is not output as an entity, mostly `date` (today, tomorrow, this weekend, etc.). Another about 22% are f misheard, and about 29% c + d (boundary or type disagreeing with the SLURP annotation).
3. The class proportions of P2 and Whisper + T2 are almost the same; the main difference is that Whisper + T2 has 2 more each in e and f.
4. **Limitations of the rules themselves** (found after seeing the results; the rules are not changed, only recorded):
   - The diagnostic normalisation replaces apostrophes with spaces, so `applebees` and `Applebee's` (`applebee s`) are not equal and are classed f. Recomputed afterwards with "delete apostrophes": in each route 1 entity (`applebees`) would become class b, and 3 class-f entities (`applebees`, `tomorrows`, `doctors appointment`) would appear in the transcript. This is only a post-hoc description and does not replace the table above.
   - Several class-c cases are **spelling errors in the SLURP gold labels themselves** (`barack obaba`, `micheal jackson`), where the model's spelling is actually correct. These rules cannot distinguish them from real boundary errors.
   - Each gold entity is compared independently with all predicted entities, so one predicted entity may explain several gold entities at once.
