English translation of c0_plan_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c0_plan_v1.md

# C0 v1: SLURP reference transcripts → frozen T2 (c0-v1; 2026-11-06; **plan and job configuration, not yet submitted, awaiting approval**)

> - Purpose: remove the "listening" problem entirely and see the upper bound of text understanding (C0 in `docs/conditions.md`).
> - This document contains only the plan and configuration. **No job submitted, no inference script written.** No downloads, no fine-tuning, no environment changes, training jobs not touched, test not touched; mix_v1, the P2/T2 prompts, slu-score-v2, binding-score-v1, existing results and verdicts are not changed.

## 1. Input
- The **SLURP reference transcripts** of the same **124 recordings** (the recordings of A and B in each mix_v1 group, the same as C1 / P2 and C7W-C1): the `sentence` of each recording in `configs/mix_v1_devel100_v2.json` (SLURP original text, lowercase, no punctuation, e.g. "do i need to bring a coat today"). No rewriting at all.
- Prompt: **the frozen T2** (`PROMPT_T2` in `scripts/qwen/qwen_omni_c1_p2t2.py`), text unchanged. The input format is exactly the same as the T2 of C7W: `build(processor, [sentence.strip() + '\n', PROMPT_T2])` (`qwen_omni_c6c7.build`), i.e. "transcript + newline + T2".
- Existing related results (reference only, not a substitute for C0): the T2 of job 22011390 was run on the reference transcripts of the 100 c1_ext recordings, of which **50** overlap with these 124. C0 is still **fully rerun** on the 124, without splicing in old outputs; afterwards, whether the outputs for these 50 are identical item by item to 22011390 is reported (a consistency observation only).

## 2. Model and decoding
- Qwen2.5-Omni-7B, commit `ae9e1690` (local snapshot, offline), bf16, sdpa, **text-only input, talker disabled** (`disable_talker()`, `return_audio=False`).
- Decoding the same as T2: greedy (`do_sample=False`), `thinker_max_new_tokens=256`.

## 3. Scoring (unchanged)
- **binding-score-v1**:
  - per recording (the same as C1 in `eval_c3_v1.py`: each recording computed over the pairs of the groups it belongs to, then averaged over recordings) — **this is the main number compared with C1(P2) = 0.694**;
  - mapped to the 200 samples (the same as C7W-C1 in `eval_c7w_f1.py`, `eval_c4_v1.score(single=True)`).
- **slu-score-v2** (`rescore_slu_v2.score_item`): strict intent accuracy, value-only P / R / F1, number of zero-entity outputs, number of parse failures.
- **Error classification**: the same classification with the rules of c1-error-taxonomy-v1 (`docs/c1_error_taxonomy_v1.md`, unchanged); e / f are judged with **the reference transcript itself** as the transcript. Since the input is the reference transcript, class f (value not in the transcript) can only come from disagreement between the gold value and the transcript text (e.g. apostrophes, annotation spelling), and is listed separately.

## 4. Job configuration (not submitted)
| Item | Value |
|---|---|
| Scripts (to be written after approval) | `scripts/qwen/qwen_omni_c0.py`, `scripts/qwen/qwen_omni_c0.sbatch`; evaluation `scripts/qwen/eval_c0_v1.py` |
| Account / partition | <SLURM_ACCOUNT> / <GPU_PARTITION> |
| Resources | 1 A100 (`--gres=gpu:a100:1`), 4 CPU, 48 GB memory |
| Time limit | 30 minutes |
| Environment | `<SHARED_ROOT>/envs/qwen_omni`; `HF_HOME=$SHARED/checkpoints/hf`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (same as existing jobs) |
| Expected duration | loading about 50–70 s; inference 124 items × about 1.04 s ≈ 2.2 minutes; about **4 minutes** in total |
| Output | `outputs/qwen_omni_c0/<jobid>/C0/raw.jsonl` and `config.json` (recording commit, uncommitted dependency files, sha256, node, GPU model, per-item time); log `outputs/logs/qwen_omni_c0_<jobid>.out`; evaluation results `outputs/c0_v1_eval/<jobid>/` |
| New disk usage | about 1 MB |
- Procedure: first a CPU dry-run (srun, constructing inputs only, no weight loading or inference, checking the 124 items and the input format), then submit the GPU job with a clean working tree. If the job fails, only report the reason; no resubmission with a changed configuration.

## 5. Interpretation written before the run (empirical, point estimates only)
Let Δ = C0 − C1(P2), both using **per-recording** target-any (C1(P2) = 0.6935, job 22014089).
1. Δ ≤ 0.05, **and** b + c + d in check A account for ≥ 50% of P2's missed entities → **"mainly an annotation-format problem"**;
2. Δ ≥ 0.15 → **"the audio path has a clear loss"**;
3. otherwise → **"both"**.
- Note (a fact, not a prediction): check A has already been run, and P2's b + c + d is 17 / 58 = **29.3%**, below 50%. Under this rule, therefore, **interpretation 1 cannot occur** whatever the C0 result; C0 can only fall into interpretation 2 or 3. The rule is not changed because of this.
- The interpretation applies only to these 124 devel recordings and one run.

---

## Results (2026-11-06; under the rules written in advance above; the rules were not changed after the results came out)
### Run record
- After approval, `scripts/qwen/qwen_omni_c0.py`, `qwen_omni_c0.sbatch` and `eval_c0_v1.py` were written (commit `c8596ee`).
- CPU dry-run: srun 22023431 (<CPU_PARTITION>, <NODE>), all 124 inputs constructed successfully, `DRY_RUN_PASS`, no uncommitted dependency files.
- GPU job 22023448: <SLURM_ACCOUNT> / <GPU_PARTITION>, 1 A100 (NVIDIA A100-PCIE-40GB, driver 610.43.02), 4 CPU, 48 GB, 30 minutes; node <NODE>, 2026-11-06 14:35:04–14:39:39 (4 min 35 s), exit code 0, MaxRSS 26.1 GB. The commit recorded by the job is `c8596ee`, with no uncommitted dependency files. Loading 49.6 s, inference 141.7 s (1.13 s per item on average). Queue time about 1 hour.
- Output: `outputs/qwen_omni_c0/22023448/C0/raw.jsonl` (124 items); evaluation: `envs/qwen_omni/bin/python -I -B scripts/qwen/eval_c0_v1.py ...`, run on the login node, reading raw outputs only, results in `outputs/c0_v1_eval/22023448/`.

### Main results
| | C1 / P2 (audio) | C7W-C1 (Whisper + T2) | **C0 (reference transcripts + T2)** |
|---|---|---|---|
| target-any (per recording) | 0.694 | 0.653 | **0.782** |
| target-any (200 samples) | 0.695 | 0.650 | 0.810 |
| All target values hit (per recording) | 0.571 | — | 0.668 |
| Strict intent accuracy | 0.766 | — | 0.734 |
| Value-only F1 (slu-score-v2) | 0.642 | — | 0.731 |
- **C0 − C1(P2) = 0.782 − 0.694 = +0.089** (per recording, 0.08871).
- Other: value-only P / R 0.790 / 0.681; 2 zero-entity outputs; 1 parse failure (`audio-1497270747`, whose reference transcript is the single word "indie"; the model replied that it "could not extract the meaning of the command", which also counts as an abstention); interferer-only on the 200 samples is 0.
- The intent accuracy and value-only F1 of C1 / P2 come from `docs/c3_results_v1.md`; these two items for C7W-C1 have not been computed on the 124 under the same basis and are left blank in the table.

### Error classification (c1-error-taxonomy-v1 rules unchanged; e / f of C0 judged with the reference transcripts)
| Class | C1 / P2 | Whisper + T2 | **C0** |
|---|---|---|---|
| Hits / misses (144 in total) | 86 / 58 | 81 / 63 | **98 / 46** |
| a. parse failure | 0 | 0 | 1 (2.2%) |
| b. different spelling | 0 | 0 | 0 |
| c. different boundary | 11 (19.0%) | 12 (19.0%) | 11 (23.9%) |
| d. wrong type | 6 (10.3%) | 6 (9.5%) | 7 (15.2%) |
| e. omission | 28 (48.3%) | 30 (47.6%) | **27 (58.7%)** |
| f. misheard | 13 (22.4%) | 15 (23.8%) | **0** |
- Of the 27 omissions of C0, 15 are `date`; the rest are place_name 3, event_name 2, definition_word 2, personal_info 2, and news_topic, game_name, business_type 1 each.
- The class c and d examples of C0 are essentially the same as for P2, e.g. `house_place "bathroom"` → `device_type "upstairs bathroom light"`, `podcast_descriptor "next podcast episode"` → `podcast_name "next podcast"`.
- Class f is 0: when the input is the reference transcript, "mishearing" disappears; the individual cases caused by apostrophe problems in class f before (e.g. `applebees`) also do not appear here.

### The 50 items overlapping with 22011390
- For the 50 items, the input prompts (`prompt_text`) are **identical item by item**, and so are the raw outputs (`raw_output`) (50 / 50).

### Interpretation (under the rules written in advance in §5)
- Δ = +0.089; in check A, P2's b + c + d account for 29.3% of misses.
- Rule 1 is not satisfied (b + c + d < 50%), rule 2 is not satisfied (Δ < 0.15) → **interpretation: both**.
- Applies only to these 124 devel recordings and one run; the thresholds are empirical, and no significance test was done.

### Observations (description only)
- After removing "listening", misses drop from 58 to 46; the 12 fewer correspond roughly to P2's 13 class-f items (misheard).
- Of the remaining 46, the numbers of omissions (27) and of boundary or type disagreements with the SLURP annotation (18) are similar under all three inputs. **Even with the reference transcript as input, about 59% of T2's misses are values present in the transcript that were not extracted**, so this part has nothing to do with "listening".
- The intent accuracy of C0 (0.734) is lower than that of C1 / P2 (0.766); the reason was not investigated.
