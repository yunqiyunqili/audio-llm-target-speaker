English translation of c1_semantic_plan.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c1_semantic_plan.md

# C1 semantic test plan (Qwen2.5-Omni-7B, structured intent + entities; run: job 22010083)

## Purpose
On clean close-talk speech (C1, original 16 kHz), confirm that Qwen can output intents and entities in the specified format, and get the parsing and normalisation rules working. Only 20 items are used; **not representative of overall performance**, and test is not used.

## Samples (list generated; no audio generated)
- Script `scripts/slurp/select_c1_semantic_devel.py`, srun (<CPU_PARTITION> / 1 CPU / 4 GB); the list is in `configs/c1_semantic_devel20.json`.
- Candidate range: basis A, official devel sentences × devel speakers (the speaker scheme with test 40% / devel 20%, seed 2026), excluding sentences whose text is the same as a test text. 146 recordings and 10 speakers in total. Whatever proportion is eventually held out from train, this part belongs to devel, so these 20 items are not affected by the choice of proportion.
- Sampling: candidates sorted by file name, shuffled with `random.Random(2026)`, then selected greedily in rounds under "at most n per intent, at most m per speaker" limits, in turn (1,1), (1,2), (2,2), unlimited, until 20 items are selected, with no repeated slurp_id. Result: 20 items covering 18 intents and 10 speakers, durations 1.6–7.5 s, each with at least 1 entity.
- Intent labels use `scenario_action`. The reason is that the `intent` field of the official jsonl disagrees with scenario_action for 229 sentences, e.g. values such as `query`, `music`, `set` lacking the scenario prefix; counted as scenario_action, the official train has 60 intents.

## Prompt
- system: Qwen's default system prompt, the same as for single-item inference.
- user: one audio clip, plus the text below. The label inventory comes from the official train: 60 intents, 55 entity types, stored in `configs/slurp_label_inventory_train.json`; the whole text is 2,043 characters.
```
This audio is a spoken command to a voice assistant. Return ONLY one JSON object, with no other text, in exactly this format:
{"transcript": "<the words spoken>", "intent": "<one intent label>", "entities": [{"type": "<one entity type>", "value": "<the entity words as spoken>"}]}
Use "entities": [] if there are none.
Allowed intent labels: <60 个 scenario_action，逗号分隔>
Allowed entity types: <55 个类型，逗号分隔>
```
- Decoding: greedy (do_sample=False), max_new_tokens=256, batch=1, text output only (talker disabled).

## Output format and parsing rules (`scripts/qwen/slu_normalize.py: parse`)
1. Remove the ```json code-block markers; take the content between the first `{` and the last `}` and parse it with `json.loads`.
2. Four statuses: `ok`, `no_json` (no JSON found), `bad_json` (cannot be parsed), `bad_schema` (not an object, or entities not a list). All except `ok` count as intent errors with empty entities, and each is counted separately.
3. Fields: `transcript` and `intent` are recorded as empty strings when not strings; in `entities` only items whose type and value are both strings are kept, and discarded items are counted separately; type is lowercased.
4. If the output intent label is not among the 60 labels, it counts as an error and is counted separately (`intent_invalid_label`), with no approximate matching.

## Normalisation rules (`scripts/qwen/slu_normalize.py: normalize`; applied equally to reference and output)
1. NFKC, lowercase, curly quotes to straight quotes; `a.m.`/`p.m.` to `am`/`pm`.
2. `%` to " percent", `&` to " and ", hyphens and slashes to spaces.
3. Times H:MM to English words: `5:30` → five thirty, `5:00` → five, `5:05` → five oh five.
4. Ordinals: `21st` → twenty first. Integers 0–9999 to cardinal words: `5` → five. Decimals: `2.5` → two point five.
5. Merge words split at an apostrophe: `derek 's` → derek's (SLURP entity values are formed by joining tokens, so this spelling occurs); delete all punctuation except apostrophes; collapse extra spaces.
- Known limitations, not handled:
  - years are read as cardinals, `1990` → one thousand nine hundred ninety, unlike the spoken "nineteen ninety".
  - `R&B` → "r and b", unlike the SLURP reference value "r n b".
  - "grass market" and "grassmarket" count as correct only in the secondary "ignore spaces" metric.
- Self-test: all 11 fixed cases of `python -I scripts/qwen/slu_normalize.py` are OK (run on the login node on 2026-11-04, no data involved).

## Metrics (`scripts/qwen/qwen_omni_c1_semantic.py`)
- Intent accuracy: correct only if exactly identical to scenario_action.
- Entities: (type, normalised value) matched as multisets, computing P, R, F1; an F1 ignoring spaces inside values is also reported.
- Transcript WER: edit distance after normalising both the reference sentence and the output transcript.
- Also recorded: counts of each parse status, number of intents outside the label inventory, per-item inference time and number of generated tokens, loading time, peak GPU memory. The full raw output of each item is saved in `raw.jsonl`.
- **Not implemented**: SLURP's official SLU-F1 (soft entity matching based on word-level and character-level edit distance).

## Job configuration and expected usage (written before submission; measurements at the end)
- Script `scripts/qwen/qwen_omni_c1_semantic.sbatch` calls `scripts/qwen/qwen_omni_c1_semantic.py`. `sbatch --test-only` passes, expected start 2026-11-05 00:22, node <NODE>.
- Configuration the same as single-item inference: <SLURM_ACCOUNT> / <GPU_PARTITION> / 1×A100 / 4 CPU / 48 GB / 30 min. Offline mode, loading from the local snapshot directory (commit ae9e1690).
- Expected usage, based on the measurements of job 22003339:
  - loading about 47 s, peak GPU memory about 20.8 GiB, process memory about 26 GB.
  - The prompt is about 2 KB characters longer than for single-item inference, and the output is at most 256 tokens. In single-item inference the second item took 0.33 s to generate 11 tokens; this time each item may take from a few seconds to over ten seconds, and 20 items are estimated at no more than 5 minutes. This is only an estimate; measurements take precedence.
  - Output files raw.jsonl and summary.json in `outputs/qwen_omni_c1_semantic/<jobid>/`, estimated below 100 KB; no audio written.

## Measured results (job 22010083; only 20 items, not representative of overall performance; the prompt was not changed on this basis)
- Run: <NODE>, 2026-11-05 00:45:54–00:48:50 (2 min 56 s), exit code 0, MaxRSS 26.4 GB. Loading 56.0 s, loading peak GPU memory 20.83 GiB, inference peak 16.93 GiB; inference of the 20 items 27.5 s in total (0.74–2.41 s each).
- Output: `outputs/qwen_omni_c1_semantic/22010083/{raw.jsonl,summary.json}`; offline rescoring (reads only raw.jsonl, run on the login node, no GPU): `rescore_summary.json`, `rescore_errors.jsonl`.
- Format: 20/20 parsed as ok; 0 discarded entities; 0 intents outside the inventory; 0 reaching max_new_tokens; 1 entity type outside the inventory (`location`).

| Metric | Strict | Relaxed |
|---|---|---|
| Intent | 14/20 = 0.70 | scenario prefix only: 16/20 = 0.80 |
| Entity F1 (hyp 22, ref 30) | TP 10, P 0.455, R 0.333, F1 0.385 | value only: TP 15, F1 0.577; same type with a shared word: TP 12, F1 0.462; ignoring spaces: TP 11, F1 0.423 |
| Transcript WER (after normalisation) | 8/154 = 0.052 | — |

- 17/20 samples have errors. Automatic error types (counted per sample, several classes per item possible): entity_missing 13, intent 6, transcript 5, entity_spurious 5, entity_type 5, entity_boundary 1, parse 0. The reference sentence, reference labels, raw output and error types of each item are in `rescore_errors.jsonl`.

## C1 extension: P0 / P1 / T0 (job 22010997; 100 devel items; scoring slu-score-v2)
- Samples `configs/c1_ext_devel100.json`; script `scripts/qwen/qwen_omni_c1_ext.py` + `.sbatch`; <NODE>, 2026-11-05 09:54:24–10:01:46 (7 min 22 s), exit code 0, MaxRSS 26.5 GB; loading 49.4 s, loading peak 20.83 GiB.
- Outputs `outputs/qwen_omni_c1_ext/22010997/{P0,P1,T0}/` (raw.jsonl with the full prompt of each item; rescore_v2_*), `condition_diff.jsonl`, `condition_diff_summary.json`.

| | P0 audio | P1 audio + 3 examples | T0 text |
|---|---|---|---|
| Parsed ok | 100 | 99 (1 no_json) | 99 (1 no_json) |
| Intent strict / scenario prefix | 77 / 81 | 72 / 76 | 71 / 76 |
| Strict entity F1 (TP, hyp/ref=128) | 0.362 (36, 71) | 0.333 (32, 64) | 0.398 (40, 73) |
| Value-only F1 / word-overlap F1 | 0.452 / 0.452 | 0.458 / 0.427 | 0.527 / 0.448 |
| Strict full semantics correct | 19 | 16 | 17 |
| WER (v2 / v1 normalisation) | 36/622=0.058 / 0.058 | 38/622=0.061 / 0.061 | — |
| Intent not in inventory / entity type not in inventory | 3 / 4 | 1 / 5 | 5 / 5 |
| Total inference time / inference peak GPU memory | 106.1 s / 16.93 GiB | 102.6 s / 16.98 GiB | 101.3 s / 16.84 GiB |

Only 100 devel items, a single run, not representative of overall performance; no prompt was selected on this basis.
