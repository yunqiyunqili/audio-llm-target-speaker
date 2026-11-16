English translation of fsc_extension_plan_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：fsc_extension_plan_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# FSC extension plan v1 (approved 2026-11-12; freeze point = the commit that adds this file, `configs/fsc_mapping_v1.json` and `scripts/fsc/score_extension_v1.py`)

> - Purpose: repeat the SLURP test primary comparison (audio LoRA − Whisper + T2-LoRA) on another dataset, with a sample size large enough to give conclusive intervals.
> - **FSC official test is not used**; SLURP test is not run again.
> - Models, prompts, front end, stream selection, abstention and checkpoints all follow the SLURP test frozen configuration (`docs/slurp_test_freeze_v1.md` §1); no retraining and no tuning.
> - **After the freeze, the mapping table and scoring code are not changed.**
> - The original 64 pilot utterances are recomputed with the new mapping **as exploratory analysis only**, reported separately from the extension results, and not used to change the mapping or rules.

## 0. Approval decisions (project lead, 2026-11-12)
1. **Single-talker samples**: 1,240 recordings (248 sentences × 5 recordings, seed 2026); bootstrap with grouped resampling by sentence text.
2. **Spatial condition**: 360 groups (720 samples), no repeated RIR within a stratum, consistent with the SLURP test rules.
3. **Primary metric** target-any; Bonferroni 98.33% intervals for the three comparisons clean, reverb and spatial, as for SLURP test. intent_exact and intent_family are descriptive only.
4. **Add C0** (reference transcripts + T2 and T2-LoRA) and the original Whisper + T2 route, descriptive only.
   - The total GPU budget is separate, **12 hours**;
   - if the re-estimate exceeds 12 hours, first drop the single-talker conditions of original Whisper + T2, and state so.
5. **Speaker constraints**: no speaker in more than 25% of the groups; male groups at least 10%.
6. **Mapping table** (§2):
   - every rule states its basis: (a) label definitions, or (b) frequencies in the FSC train annotations, with counts;
   - no model output is used as a basis;
   - mute is mapped to the most common action label among FSC train sentences containing "mute";
   - wemo maps only the action, with no default object;
   - heat synonyms use only expressions that actually occur in FSC train sentences whose object is heat.
7. **The 64 pilot utterances**: recomputed with the new mapping, exploratory analysis only, reported separately from the extension results.
- **Stopping conditions**:
  - fewer than 300 usable groups after sampling;
  - any job still failing after a rerun;
  - the GPU re-estimate exceeds 12 hours and still exceeds it after dropping the single-talker conditions of original Whisper + T2.
  - Failure reruns as in SLURP §0-8: a job may be rerun once unchanged only if it produced no complete output.

## 1. Samples

### 1.1 Single-talker conditions (clean, reverb)
- **Source**: FSC valid (3,118 recordings, 10 speakers, 248 sentence texts, 31 triples; `valid_data.csv` sha256 `5ac95de1…5197`). The 64 recordings already used by the pilot (`outputs/fsc_pilot/valid64_manifest.jsonl`, sha256 `64a695df…c173`, matched by path) are excluded, leaving 3,054 recordings.
- **Sampling rules** (written before the run):
  1. Sort the 248 sentence texts by text string; `rng = random.Random(2026)`.
  2. For each sentence text: sort the remaining recordings by path and shuffle them with rng; pick recordings one by one in shuffled order, choosing each speaker only once, until 5 recordings are chosen.
     - If the remaining recordings cover fewer than 5 different speakers, fill up in the same order. Check: every sentence text has at least 6 remaining recordings and at least 5 different speakers, so the quota can always be filled.
  3. In total **248 × 5 = 1,240 recordings**. The distribution over the 31 triples is proportional to the text distribution of valid.
- **Text overlap with SLURP train**: the pilot audit found 349 recordings (25 sentence texts) in valid whose text is identical to SLURP train texts, plus 15 near-duplicates (`outputs/fsc_pilot/audit.json`).
  - These recordings are **not excluded**; they are flagged item by item;
  - a sensitivity result is also reported: recomputed after removing these 25 sentence texts, descriptive only.
- **Rationale for the sample size**:
  - In the pilot, the per-item standard deviation of the Audio − T2-LoRA paired difference was 0.33 (clean), 0.37 (reverb), 0.36 (spatial). This is used only to estimate variance, not for the mapping. On SLURP test, C1 uses 1,183 recordings and the 98.33% interval half-width is about 0.027.
  - The bootstrap is grouped by **text** (the equivalent of SLURP's slurp_id), and multiple recordings of the same text are correlated. With per-item standard deviation σ = 0.32 to 0.40 and within-text correlation ρ = 0 to 0.3, the expected half-widths of the 98.33% interval are:

| σ | ρ | 1,240 recordings (5 per sentence) | 3,054 recordings (all) |
|---|---|---|---|
| 0.32 | 0 / 0.1 / 0.3 | 0.022 / 0.026 / 0.032 | 0.014 / 0.020 / 0.029 |
| 0.40 | 0 / 0.1 / 0.3 | 0.027 / 0.032 / 0.040 | 0.017 / 0.025 / 0.036 |

  - FSC has only 248 sentence texts; when the within-text correlation is high, the benefit of adding recordings diminishes quickly. 1,240 recordings give about ±0.02–0.03 at ρ ≤ 0.1, with only 40% of all recordings.
  - **1,240 recordings are adopted** (§0-1).

### 1.2 Two-talker spatial condition (spatial)
- **Pairing**: drawn from the 1,240 recordings above with mix_v1-style rules:
  - same gender (FSC valid has 7 female and 3 male speakers), different speakers, duration difference ≤ 1 s;
  - the target and interferer triple values (action, object, location, excluding `none`) each have distinguishable values, i.e. D_T and D_I are both non-empty;
  - each recording used at most 3 times.
- **Sampling**: seed 2026. Sort by (path_a, path_b), shuffle, then select N groups by rank with 0/1 integer programming; if infeasible, decrease in multiples of 4 (as for SLURP test).
- **Speaker constraints**: FSC has only 10 speakers, so SLURP's "≤ 30 groups per speaker" cannot be met proportionally. Changed to "each speaker ≤ 25% of the groups" and male groups ≥ 10% of the groups. Approved (§0-5).
- **RIRs**: SMS-WSJ **test_eval92** (consistent with SLURP test; not cv_dev93, which was used for tuning on devel), equal across the four angle strata, no repeated RIR sample within the FSC set. Repeating samples already used by SLURP test is allowed (RIRs are simulated draws and involve no tuning).
- **Mixing**: offsets, early RIR, level, SIR 0 dB and SNR U(20, 30) dB are the same as in `mix_slurp_v1.py`; swapping the target within each group gives two samples; the query direction is θ1.
- **Number of groups**:
  - SLURP test C5 uses 400 samples, with a 98.33% half-width of about 0.048. Reaching ±0.03 would need about 1,000 samples (500 groups).
  - But the [15,30) stratum of test_eval92 has only 94 samples, so **without repeated RIRs and with four equal strata, at most 94 × 4 = 376 groups**.
  - **360 groups are adopted (720 samples, 90 groups per stratum)**, no repeated RIR within a stratum (§0-2); expected half-width about 0.035. If the integer program is infeasible, decrease by 4; stop if fewer than 300 groups.

## 2. Mapping table (frozen file `configs/fsc_mapping_v1.json`; executed by `predict()` in `scripts/fsc/score_extension_v1.py`)

- **Note**: the author of the mapping table had seen part of the pilot's model outputs during pilot scoring and spot checks. The rules of the mapping table are based **only** on the two kinds of source below; no model output was used as a basis, and nothing was adjusted according to model outputs:
  - **(a) Label definitions**: the names of intents and entity types in the SLURP training-set label inventory `configs/slurp_label_inventory_train.json` (60 intents, 55 entity types, sha256 `a3e86457fafa4aaf60f28654f3778a278b29132068977285b3996c46773880e9`), and FSC's 31 (action, object, location) triples;
  - **(b) Frequencies in the FSC train annotations**: `train_data.csv`, 23,132 records in total. Count = the number of train records whose annotated text contains the word, and their distribution over labels.
- **Scope**: applies only to the parsed `intent` and `entities` of the model output. Each entity value is first passed through `normalize_v2`, then split into whole words by `toks` and matched as whole words within the value; the first matching entity in output order is taken. **The transcript field is not used.**
- **Unit tests**: 15 hand-constructed outputs (not model outputs) all give the expected triples under the rules.

### 2.1 Object
- **Keyword rule**: a word is accepted if at least 95% of its FSC train records fall in the same non-none object class, with at least 20 records, and it is the name of the object itself or of its degree. Verbs and function words are not accepted; the full list with counts is in `object_keywords_rejected` in the json.

| FSC object | Keywords | Basis | Counts (train records containing the word / records falling on this object) |
|---|---|---|---|
| lights | lights | a + b | lights 4,794 / 4,794. The singular light occurs 0 times in train and is not accepted |
| lamp | lamp | a + b | 792 / 792 |
| heat | heat, temperature, heating, hotter, cooler | b (only expressions that actually occur in sentences whose object is heat) | heat 2,946 / 2,946; temperature 2,805 / 2,805; heating 1,322 / 1,322; hotter 92 / 92; cooler 90 / 90 |
| volume | volume, sound, louder, loud, quiet, quieter, audio, softer | a + b | 2,203; 647; 492; 476; 278; 192; 190; 100 (each 100% on volume) |
| juice / newspaper / socks / shoes | the same nouns | a + b | 466 / 551 / 538 / 536 (each 100%) |
| Chinese / English / German / Korean | chinese / english / german / korean | a + b | 449 / 349 / 342 / 345 (each 100%) |
| music | (no keyword) | b | the word music: 988 records fall on music and 198 on volume, 83% < 95%, not accepted. The object music comes only from the default object of play_music |
- If one entity value contains keywords for both lights and lamp, take lamp. Basis a: FSC has no triple containing both.
- If nothing matches, use the intent's default object (§2.3); if there is none, `none`.

### 2.2 Location
| FSC location | Keywords | Basis | Counts |
|---|---|---|---|
| kitchen | kitchen | a + b | 2,608 / 2,608 |
| bedroom | bedroom | a + b | 2,412 / 2,412 |
| washroom | washroom, bathroom | a + b | washroom 2,211 / 2,211; bathroom 1,751 / 1,751 (FSC label washroom in all cases) |
| none | nothing matches | a | — |

### 2.3 SLURP intent → FSC action (and default object)
| SLURP intent | action | Default object | Basis |
|---|---|---|---|
| iot_hue_lighton / lightoff | activate / deactivate | lights | a |
| iot_hue_lightup / lightdim | increase / decrease | lights | a |
| iot_hue_lightchange | unmappable | lights | a (colour change; FSC has no corresponding action) |
| iot_wemo_on / iot_wemo_off | activate / deactivate | **none** | a; per the decision, only the action is mapped |
| play_music | activate | music | a + b (train records containing play: 285 / 285 are activate music) |
| play_radio / play_podcasts / play_audiobook | activate | none | a |
| audio_volume_up / audio_volume_down | increase / decrease | volume | a |
| **audio_volume_mute** | **decrease** | volume | **b: 99 train records contain mute; action 99 / 99 decrease, object 99 / 99 volume** |
| audio_volume_other, music_settings | unmappable | volume / music | a |
| the other 44 intents | unmappable | none | a |

### 2.4 Action determined by the object (overrides the result of §2.3)
- object ∈ {juice, newspaper, socks, shoes} → **bring**. Basis a: among the 31 triples these objects co-occur only with bring.
- object ∈ {Chinese, English, German, Korean} → **change language**. Basis a, as above.
- Any entity value containing the whole word language (2,298 / 2,298) or languages (92 / 92) → **change language**; if there is then no language name, object = `none`. Basis b.
- **Unmappable** actions count as errors and stay in the denominator.

### 2.5 Metrics
- **Predicted triple** = (action, object, location); `none` is also compared as a value.
- **target-any (primary metric)**: among the gold slots that are not `none`, the predicted value is equal in at least one slot.
- **intent_exact** (descriptive): all three slots equal.
- **intent_family** (descriptive): action and object equal, location ignored.
- **Per-slot accuracy**, proportion of unmappable actions, proportion of object = `none`, number of parse failures: all reported per route and per condition.
- **Two-talker conditions**: binding-score-v1's `judge` applied to the (slot=value) multisets, D_T = T − I, D_I = I − T; target-only, interferer-only and pair flip are reported.

## 3. Primary comparison and statistics (same as SLURP test)
- **Direction**: audio LoRA − Whisper + T2-LoRA, paired on the same audio.
- **Three primary comparisons**: clean (corresponding to C1), reverb (C1-R), spatial selected stream (C5, equal-weight basis).
- **Bonferroni**: each is judged significant only if its 98.33% interval lower bound is > 0; 95% intervals are also reported.
- **Primary metric**: target-any (§2.5), consistent with SLURP. intent_exact and intent_family are descriptive only (§0-3).
- **bootstrap**: B = 2000, `default_rng(2026)`.
  - single-talker conditions grouped by text;
  - spatial condition resampled by group, stratified by angle stratum, the two swaps drawn together.
- **Descriptive only**:
  - the improvement of both LoRAs over their respective baselines (original Qwen + P2, original Whisper + T2);
  - c6-real main conditions (8-sector centres, σ10, abstention A τ = 17°, abstention B), natural weighting with the cv_dev93 distribution;
  - interferer risk and the diagnostics split by routing (as in Section 6 of the SLURP test report);
  - sensitivity analysis for text overlap;
  - exploratory recomputation of the 64 pilot utterances.
- After the results come out, no rule, mapping or checkpoint is adjusted. Failure reruns as in SLURP §0-8.

## 4. Conditions and routes
| Condition | Input | Routes | Samples |
|---|---|---|---|
| clean | FSC original 16 kHz (same chain as SLURP C1, no level adjustment) | original Qwen + P2; audio LoRA + P2; original Whisper + T2; Whisper + T2-LoRA | 1,240 |
| reverb | `reverb_single.reverb()` chain, test_eval92 RIRs, one distinct sample per recording | same as above | 1,240 |
| spatial | 6-channel mixture → SpatialNet ep100 → v3 stream selection → chain8 (polarity applied only once) | same as above; inference on both the selected and the other stream | 720 samples (360 groups) |
| c6-real main conditions | reuse the two spatial output streams; change the query direction and stream selection on CPU | same as above | same as spatial |
- Original Whisper + T2 is a route the pilot did not have; it is added to be consistent with "both LoRAs relative to their own baselines" in the SLURP report.
- Preprocessing always uses the `scripts/fsc/pilot_audio.prepare()` chain already checked in the pilot.
- C0 (FSC reference transcripts → T2 / T2-LoRA) is descriptive only, about 1.1 GPU h (§0-4).

## 5. Job configuration and GPU estimate
- **GPU jobs**: <SLURM_ACCOUNT> / <GPU_PARTITION>, 1 × A100, 4 CPU, 48 GB, 20 minutes, each shard ≤ about 14 minutes.
  - the original Qwen routes use `envs/qwen_omni`, the LoRA routes `envs/qwen_omni_lora`;
  - SpatialNet uses 32 GB.
- **CPU jobs**: <CPU_PARTITION>, 4 CPU, 16 GB, 20 minutes, for construction, stream selection and scoring.
- The working tree must be clean before submission; each job records the commit, input and output sha256, and model and adapter sha256.
- **Per-item time** (measured in the FSC pilot; FSC utterances are shorter): original P2 1.32 s, audio LoRA 2.23 s, Whisper + T2-LoRA 2.25 s (including Whisper); original Whisper + T2 estimated at 1.4 s. About 7.2 s for the four routes together; loading about 1.6 minutes per job.

| Part | Inference items | Estimated GPU time |
|---|---|---|
| clean (1,240 × 4 routes) | 4,960 | about 2.5 h |
| reverb (1,240 × 4 routes) | 4,960 | about 2.5 h |
| spatial (720 × 2 streams × 4 routes) | 5,760 | about 2.9 h |
| SpatialNet front end (720) | 720 | < 0.1 h |
| Loading overhead (about 42 jobs) | — | about 1.1 h |
| **Total** | — | **about 9.1 GPU h** |
| Optional C0 (1,240 × 2) | 2,480 | about 1.1 h |
| If the single-talker conditions used all 3,054 recordings | — | total about 16 h |

- The budget is separate, **12 GPU hours** (including C0 and original Whisper + T2). Re-estimate from measurements before execution: if above 12 h, first drop the single-talker conditions of original Whisper + T2 and state so; if still above, stop and report.

## 6. Execution order (approved)
- Code: sampling and pairing `scripts/fsc/ext_build_list.py`, reverberation `ext_make_reverb.py`, task lists `ext_tasks.py`, inference `ext_batch.py`, scoring `score_extension_v1.py`.
- Mixing, direct-path references, SpatialNet separation and stream selection reuse the scripts already checked for SLURP test: `split_cfg.py` gets a new `fsc_ext` branch, with the default devel and test branches unchanged; the job wrapper scripts use EVAL_SPLIT=fsc_ext.
1. **CPU, write the code and freeze (this commit)**:
   - sampling list, pairing integer program, mapping table and scoring script;
   - devel equivalence check, requiring byte-identical results where existing SLURP code is used;
   - unit tests of the mapping rules on the 64 pilot utterances;
   - commit and push; no code changes afterwards.
2. **CPU construction**: 1,240 single-talker recordings, test_eval92 reverberation, 360 mixture groups, direct-path references, with all sha256 recorded.
3. **GPU**: SpatialNet → CPU stream selection → inference (clean, reverb, both spatial streams).
4. **CPU scoring**: one-off report; plus the exploratory recomputation of the 64 pilot utterances.
