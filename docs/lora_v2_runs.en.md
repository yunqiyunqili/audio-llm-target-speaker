English translation of lora_v2_runs.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：lora_v2_runs.md

# LoRA v2 run records (executed according to `docs/lora_slu_plan_v2.md`; 2026-11-06)

> The rules are in the frozen `docs/lora_slu_plan_v2.md`; this file only records actual runs and results and does not change the rules. No training (the pilot test only measures speed and GPU memory), test is not touched, training jobs are not touched.

## 1. Environment (see plan §10)
- `envs/qwen_omni_lora`: srun 22023507 cloned it and installed peft 0.15.2 (wheel sha256 `0dfc942b…93d0a0`, matching PyPI); srun 22023514 passed the import and minimal LoRA test. `pip freeze` of the original environment is identical before and after installation.

## 2. C0 devel-428 baseline (plan §11 item 4, "now")
- Script: `scripts/qwen/qwen_omni_c0_devel428.py` / `.sbatch` (same code path as C0 v1, only the sample table changed: all 428 recordings of split_v2 devel, gold labels from `devel_recordings.csv`, matching item by item the 124 in the mix_v1 configuration).
- CPU dry-run: srun 22023516 (all 428 inputs constructed successfully).
- GPU job 22023517: <SLURM_ACCOUNT> / <GPU_PARTITION>, NVIDIA A100-PCIE-40GB, 4 CPU, 48 GB, 30 minutes; node <NODE>, 15:31:52–15:42:32 (10 min 40 s), exit code 0, MaxRSS 25.4 GB; commit `80bd3e3`, no uncommitted dependencies. Loading 44.6 s, inference 520.3 s (about 1.22 s per item).
- Evaluation: `scripts/qwen/eval_c0_devel428.py` (login node, read-only), results in `outputs/c0_devel428_eval/22023517/`.

| | All 428 | Selection half (214 recordings / 191 sentences) | Report half (214 / 191) |
|---|---|---|---|
| At least one gold value hit (single-talker, no interferer) | 0.720 | 0.715 | 0.724 |
| Hits / gold entities | 349 / 590 | 178 / 298 | 171 / 292 |
| date recall | 0.500 (64 / 128) | 0.365 | 0.631 |
| Strict intent accuracy | 0.706 | 0.668 | 0.743 |
| Value-only P / R / F1 | 0.718 / 0.592 / 0.649 | — | — |
- Error classes (c1-error-taxonomy-v1 rules, transcript = reference transcript), 241 misses over all 428: a parse failure 3, b different spelling 1, c different boundary 57, d wrong type 55, **e omission 121**, f 4. Of the 64 date misses: e 42, c 12, d 10.
- Other: 11 zero-entity outputs, 3 parse failures, 2 abstentions, transcript WER 0.0077.
- **Consistency**: for the 124 recordings overlapping with C0 v1 (job 22023448), the input prompts and raw outputs are **124 / 124 identical item by item**; different recordings of the same sentence (41 sentences) also give exactly the same outputs.
- Note: "at least one gold value hit" is the single-talker basis (binding-score-v1 value matching, D_T = all gold values, no interferer), and is **not the same metric** as the 0.782 computed by pair on the 124 mix_v1 recordings. The single-talker conditions of plan §5 did not state this basis; see "needs decision".

## 3. Step-0 pilot test (plan §7 item 2, §8)
- Script: `scripts/lora/text_lora_pilot.py` / `.sbatch` (commit `80bd3e3`). Data from train only: 2,363 sentences (deduplicated by slurp_id, shuffled with seed 2026); the intersection of sentences, slurp_ids and speakers with devel / test is 0 (script assertion). Inputs average 611 tokens (max 666), of which the answer part averages 45 tokens.
- CPU dry-run: srun 22023518. GPU job 22023519: <NODE>, A100-PCIE-40GB, 15:43:02–15:46:31 (3 min 29 s), exit code 0, MaxRSS 4.7 GB; no uncommitted dependencies.
- 112 LoRA modules (thinker language model only), 10,092,544 trainable parameters; loading 21.2 s, GPU memory after loading 16.68 GiB.

| | Median per micro-step | Per sample | Peak GPU memory allocated / reserved |
|---|---|---|---|
| micro-batch 1 (200 steps) | 0.260 s | 0.260 s | 17.50 / 21.03 GiB |
| micro-batch 4 (40 steps) | 0.793 s | 0.198 s | 19.61 / 21.78 GiB |
- The first step has a warm-up of about 3 s, excluded from the median. Peak GPU memory is far below the 38 GB stopping line.
- Loss: very low from the start (micro-step loss 0.016–0.23), because the original model can already predict most answer tokens (repeating the transcript, JSON format). Averaged per optimizer step (every 16 items): the 12 steps of the micro-batch 1 phase fluctuate between 0.066 and 0.158, the 10 steps of the micro-batch 4 phase between 0.042 and 0.121. **With so few steps no clear downward trend is visible**, and the pilot test was not designed for that.
- Saving: adapter `adapter_model.safetensors` 40.4 MB (fp32), optimizer state 80.9 MB.
- Format check (3 sentences of the devel selection half, after 22 optimizer steps): 3 / 3 parsed successfully, format normal.
- **Re-estimate from measurements**: for step 0, 3 epochs per hyperparameter setting (2,363 × 3 = 7,089 items) take about **0.39 h** (micro-batch 4) to 0.51 h (micro-batch 1); about 1.2–1.5 h for all 3 settings. C0-428 evaluation of each checkpoint takes about 10 minutes (1.22 s per item + loading). The 2–3.5 h estimated in plan §8 is conservative.
- **Not tested**: segmented resumption (resuming training in a new job after saving). A single setting takes about 0.4–0.5 h, more than 30 minutes, so it needs 2 segments, and resumption must be verified after the first segment of formal training.

## 4. Step-0 formal training: implementation conventions (2026-11-06, **written before any formal training**; supplementing details not written in revision 1)
- Scripts: `scripts/lora/train_text_lora.py` / `.sbatch` (training), `scripts/lora/eval_text_lora_ckpt.py` / `.sbatch` (checkpoint inference). Checkpoints and resume states go to `<SHARED_ROOT>/checkpoints/lora_v2/step0/lr<lr>/` (not in the repository).
- Steps: in each epoch the 2,363 sentences are shuffled with `random.Random(2026 + epoch)`, 4 sentences per micro-batch, one optimizer update every 4 micro-batches (148 steps per epoch, the last step with only 11 sentences); 444 steps over 3 epochs. Warmup 13 steps (3%), then cosine decay to 0. Checkpoints at steps 74, 148, 222, 296, 370, 444 (0.5–3 epochs). Segment 1 is steps 1–222, segment 2 steps 223–444.
- Resume verification: segment 1 saves the resume state at step 222, then runs 2 extra "reference steps" (not saved), recording learning rate, batches, loss and a LoRA parameter checksum. At the start of segment 2, **exact checks** of the global step, the learning rate of the next two steps, the optimizer state (checksum and per-parameter step counts) and the data order of the next two steps; if any item does not match, exit (exit code 3) without training. Then rerun those 2 steps and compare loss and parameter checksum with the reference values, **report only** (GPU computation may have tiny non-determinism).
- Order of settings: the first setting is learning rate 5e-5; the other two settings are submitted only after the resume verification of segment 2 of the first setting passes.
- **Selection**: all checkpoints are run and scored only on the **selection half** (191 sentences / 214 recordings); the best is chosen by "at least one gold value hit", ties going to fewer steps and then to the smaller learning rate. **The report half is run only once, for the best checkpoint**, for "also reported".
- Inference: each sentence is generated only once, and the result is used for all recordings of that sentence (in C0-428, different recordings of the same sentence give exactly the same output).
- Exceptions to the gate (§7 item 4):
  - "training loss did not decrease": in the setting containing the best checkpoint, the mean training loss of the last 37 steps (0.25 epoch) is not below the mean of the first 37 steps;
  - "output format broke": the parse-failure rate of the best checkpoint on the selection half is > 2% (same as the §6 threshold).
- Gate bootstrap: on the selection half, the per-recording difference = (best checkpoint hit or not) − (C0-428 baseline hit or not); grouped resampling of the 191 sentences by slurp_id, B = 2000, `np.random.default_rng(2026)`, the 2.5% quantile as the lower bound. Note: the best checkpoint is selected on the same selection half, so this lower bound is optimistic.

## 5. CPU evaluation scripts for the gate and the report (2026-11-06, **written before any checkpoint results**)
- Timing check (correction): while writing and dry-running, 22023808 was queued; it started running at 16:30:14, and when the script was committed (`2bff94f`, 16:31:16) it had been running for about 1 minute, but the checkpoint directory `.../step0/lr5e-5/` still did not exist after the commit (16:31:21), and neither the log nor the training output had been looked at.
- Script: `scripts/lora/eval_text_lora_gate.py` (CPU, run read-only on the login node; metric functions directly reuse `summary` of `scripts/qwen/eval_c0_devel428.py`, `judge` / `vals` of binding-score-v1, `score_item` of slu-score-v2, and the 124-recording pair basis of `eval_c0_v1.py`, all unchanged).
- `gate`: selection half only (214 recordings / 191 sentences). Checkpoints are ranked by "at least one gold value hit", ties to fewer steps, then the smaller learning rate. bootstrap: per-recording difference = (candidate hit or not) − (C0-428 hit or not); the slurp_ids of the selection half sorted ascending, grouped resampling with `np.random.default_rng(2026).integers(0, n, size=(2000, n))`; the statistic is the mean difference weighted by recordings within the resampled sentences; the 95% interval is `np.quantile(..., [0.025, 0.975])` (linear interpolation); passes only if the lower bound is > 0. Exceptions: in the `log.jsonl` of the setting containing the best checkpoint, the mean loss of steps 408–444 ≥ that of steps 1–37 (when a step has several records, the last one is taken); or parse failures on the selection half (slu-score-v2 status ≠ ok, including abstentions) > 2%. If there is an exception, report the reason first.
- `report`: the selection-half and report-half outputs of the best checkpoint are combined into 428 items; for the selection half / report half / all, it gives the candidate's and baseline's "at least one gold value hit", date recall, c1-error-taxonomy-v1 class counts, intent accuracy and parse failures, the pair-based target-any on the 124 mix_v1 recordings (averaged by recording, compared with the 0.782 of C0 v1), and the paired bootstrap difference by the same method (report only).
- **Dry-run** (login node, the output of baseline 22023517 used as the candidate and compared with itself; output in a temporary directory, not in the repository):
  - `gate`: all 3 candidates are 0.715, step 74 / 5e-5 chosen by the rule; difference 0.0, interval [0.0, 0.0]; the exception check was exercised with an artificial decreasing-loss log (the actual gate uses the real `log.jsonl`); parse failures on the selection half 1 / 214.
  - `report`: the differences for the selection half / report half / all are all 0, intervals all [0, 0]; the baseline numbers agree with §2 (0.715 / 0.724 / 0.720; date 0.365 / 0.631 / 0.500; classes over all 428: a 3, b 1, c 57, d 55, e 121, f 4); 124 by pair: selection half 63 recordings 0.762, report half 61 recordings 0.803, all **0.782** (consistent with C0 v1).

## 6. Decision record: submitting segment 1 of the other two settings early (2026-11-06 16:5x EST, project lead; written before the resume verification)
- 22023858 (5e-5 segment 2, including the resume verification) was queued (Priority). The project lead decided: **submit segment 1 of the 1e-4 and 2e-4 settings now, without waiting for the resume verification** (changing the order "the other two settings are submitted only after the resume verification passes" of §4; the rest of the configuration is unchanged).
- If the resume verification later fails: these two segment-1 runs are **void** (not used for selection or the gate); per the approved fallback, each setting is rerun as a single segment with a 60-minute job limit, and the reason is recorded in this file.
- When reporting, also give the means of the first 37 and last 37 steps of the total loss and of the entity-token loss (observation only).

## 7. Step-0 formal training: run records (3 settings × 2 segments, all completed)
- All <SLURM_ACCOUNT> / <GPU_PARTITION>, NVIDIA A100-PCIE-40GB, 4 CPU, 48 GB, 30 minutes; exit code 0; peak GPU memory allocated 20.83 GiB (the `SEGMENT_DONE` line of each segment log); MaxRSS about 25–26 GB; training script sha256 `78aba82c…` identical in all segments, no uncommitted dependencies.

| Job | Setting / segment | Node | Start–end | Duration | commit (read at job start) |
|---|---|---|---|---|---|
| 22023808 | 5e-5 / 1 (steps 1–222) | <NODE> | 16:30:14–16:45:12 | 14:58 | `2bff94f` (see the timing check in §5) |
| 22023858 | 5e-5 / 2 (steps 223–444) | <NODE> | 18:09:57–18:24:13 | 14:16 | `51081b3` |
| 22023864 | 1e-4 / 1 | <NODE> | 18:24:38–18:38:01 | 13:23 | `51081b3` |
| 22023865 | 2e-4 / 1 | <NODE> | 18:28:10–18:42:38 | 14:28 | `51081b3` |
| 22024003 | 1e-4 / 2 | <NODE> | 19:18:14–19:32:41 | 14:27 | `51081b3` |
| 22024004 | 2e-4 / 2 | <NODE> | 19:19:15–19:32:38 | 13:23 | `51081b3` |
- About 3.2–3.4 s per optimizer step. Each of the three settings has `ckpt_step74/148/222/296/370/444`; each `log.jsonl` has 444 records (one for each of steps 1–444), with no non-finite loss.
- **Resume verification** (done for all 3 settings): all 5 exact checks pass; after rerunning the two reference steps (223, 224), the loss and LoRA parameter checksum are **exactly identical** to the reference values (relative difference 0.0). The fallback of §6, "void and switch to a single 60-minute segment", was therefore not triggered, and segment 1 of 1e-4 / 2e-4 is valid.
- Training loss (mean per step; total loss / entity-token loss, the latter for observation only):

| Setting | Steps 1–37 | Steps 186–222 | Steps 408–444 |
|---|---|---|---|
| 5e-5 | 0.114 / 0.183 | 0.033 / 0.054 | 0.030 / 0.047 |
| 1e-4 | 0.101 / 0.163 | 0.028 / 0.045 | 0.021 / 0.034 |
| 2e-4 | 0.089 / 0.144 | 0.024 / 0.039 | 0.015 / 0.024 |
- All three settings satisfy "mean of the last 37 steps below that of the first 37", i.e. the "loss did not decrease" exception of §4 does not hold (the gate script recomputes this with the same rule).
- CPU dry-run of the checkpoint inference script: srun 22024017 (<CPU_PARTITION>, <NODE>), selection half 214 recordings / 191 sentences, report half 214 recordings / 191 sentences, all inputs constructed successfully.

## 8. Selection-half checkpoint inference (18) and timeout reruns
- 6 jobs (3 checkpoints each, `eval_text_lora_ckpt.sbatch <lr> <steps> selection`, 30 minutes, commit `e0423bd`, A100-PCIE-40GB): 22024019 (5e-5 steps 74/148/222, <NODE>), 22024020 (5e-5 steps 296/370/444, <NODE>), 22024023 / 22024024 (2e-4 two batches, <NODE>) completed with exit code 0; 22024021 / 22024022 (1e-4 two batches, <NODE>) **timed out** (TIMEOUT, 30 minutes).
- **Cause of the timeout**: the two 1e-4 jobs were placed on the same node <NODE> and ran simultaneously, taking 518–649 s per checkpoint (360–380 s on other nodes); 3 checkpoints plus about 50 s of loading exceeded 30 minutes. 22024021 completed steps 74 and 148 but not step 222; 22024022 completed steps 296 and 370 but not step 444. The unfinished checkpoint directories contain only `config.json`, no `raw.jsonl`, and are not used for selection. In addition, the actual inference speed is about 1.9 s / sentence; the handoff document's estimate of 1.2 s / sentence was too low.
- **Rerun method** (approved by the project lead 2026-11-06): one job per missing checkpoint, with the rest of the configuration unchanged: `sbatch scripts/lora/eval_text_lora_ckpt.sbatch 1e-4 222 selection`, `... 1e-4 444 selection`. The 16 completed checkpoints are not rerun. Outputs of the checkpoints completed within the timed-out jobs are used as usual.
- No partial scores are looked at until all 18 checkpoints have 214 outputs; then `eval_text_lora_gate.py gate` is run. Report-half inference is likewise 1 checkpoint per job.
- Reruns: 22024071 (1e-4 step 222, <NODE>, 20:33:36–20:47:31, 13:55, 626 s), 22024072 (1e-4 step 444, <NODE>, 20:38:39–20:45:28, 6:49, 379 s), both exit code 0, A100-PCIE-40GB, commit `ec63115`, no uncommitted dependencies. Afterwards it was checked that each of the 18 checkpoints has exactly one selection-half `raw.jsonl` with 214 items.

## 9. Step-0 gate result (`eval_text_lora_gate.py gate`, login node, output `outputs/lora_text_gate/step0_selection/gate.json`)
- Selection half (214 recordings / 191 sentences), "at least one gold value hit"; C0-428 baseline 0.715:

| Setting \ step | 74 | 148 | 222 | 296 | 370 | 444 |
|---|---|---|---|---|---|---|
| 5e-5 | 0.850 | 0.888 | 0.893 | 0.897 | 0.902 | 0.902 |
| 1e-4 | 0.874 | 0.883 | 0.897 | **0.911** | 0.907 | 0.911 |
| 2e-4 | 0.888 | 0.893 | 0.897 | 0.911 | 0.902 | 0.897 |
- **Best checkpoint: 1e-4 step 296** (0.911; tied with 1e-4 step 444 and 2e-4 step 296; by the rule, fewer steps first, then the smaller learning rate).
- **Gate**: paired difference +0.196, 95% interval **[0.133, 0.264]**, lower bound > 0 → **pass**. No exception holds: the mean loss of steps 408–444 of that setting is 0.021 < 0.101 for steps 1–37; parse failures on the selection half 0 / 214. Note: the best checkpoint is selected on the same selection half, so the lower bound is optimistic (§4).
- Best checkpoint on the selection half (descriptive only): 254 / 298 gold entities hit (baseline 178); date recall 0.873 (baseline 0.365); strict intent accuracy 0.907 (baseline 0.668); classes a 0, b 1, c 20, d 5, e 15, f 3 (baseline a 1, b 1, c 34, d 28, e 53, f 3); zero-entity outputs 0, abstentions 0.
- Next step (approved): run the best checkpoint once on the report half (1 job, 1 checkpoint), then run `eval_text_lora_gate.py report`.

## 10. Preparation for steps 1 and 2: job configuration and CPU dry-run (2026-11-06, **no GPU job submitted**; awaiting the project lead's approval)
- Scripts: `scripts/lora/reverb_single.py` / `.sbatch` (single-talker full reverberation, CPU), `scripts/lora/baseline_audio.py` / `.sbatch` (P2, Whisper + T2, Whisper transcription; sharded by `--part K --nparts N`, one shard per job, Slurm limit 20 minutes). Decoding settings are the same as the frozen C1 / C7-W path (`qwen_omni_c7w_f1.py`): Qwen2.5-Omni-7B `ae9e1690` bf16, talker disabled, greedy, 256 tokens; Whisper large-v3 `06f233fe` fp16, en / transcribe, greedy, 128 tokens; T2 input is `[转写.strip() + 换行]`. devel clean audio is input as is (no level adjustment, same as C1 / W_C1).
- **Single-talker full reverberation chain** (plan §3.1 says "same as the C5 chain"; the following makes it concrete without changing the plan): 16 kHz SLURP → `resample_poly(1, 2)` → convolution with mic 0 of the SMS-WSJ RIR, aligned with `rir_start` (6 channels) of `mix_slurp_v1` and `extract_piece(·, −rir_start, len)`, tail cut to the original length → `resample_poly(2, 1)` → `qwen_omni_c4.level` (active-frame RMS −30 dBFS) → 16-bit wav. **No interferer and no noise added.** RIRs: cv_dev93 for devel, train_si284 for train (test_eval92 not read). Assignment: recordings sorted by file name, `random.Random(2026).sample(按整数排序的样例号, n)` (samples all distinct), then `randrange(2)` per item to choose h_0 / h_1.
- **CPU dry-run** (no audio written):
  - srun 22024128 (<CPU_PARTITION>, <NODE>): devel 428 items, 0.047 s each, peak 0.83, clipping 0, 428 distinct samples, T60 0.20–0.50 s, 16-bit 39.2 MB; first half of train 1,756 items, 0.048 s each, peak 0.89, clipping 0, 3,513 distinct samples.
  - srun 22024131 (<CPU_PARTITION>, <NODE>): P2 devel_clean two shards, P2 devel_rev, Whisper + T2 devel_clean / devel_rev, Whisper train_all shard 0 (clean) and shard 3 (reverberant, computed in memory) all `DRY_RUN_PASS`; P2 inputs 622–682 tokens, T2 inputs 562–579 tokens, Whisper features [1, 128, 3000].
- **Job configuration** (GPU: <SLURM_ACCOUNT> / <GPU_PARTITION>, 1 × A100, 4 CPU, 48 GB, limit 20 minutes; CPU: <SLURM_ACCOUNT> / <CPU_PARTITION>, 2 CPU, 8 GB, limit 20 minutes). Expected durations from measured speeds: P2 1.16 s / item (22014089), Whisper 0.316 s / item (W_C1), T2 1.22 s / item (22023517), loading about 1 minute; "slow node" at 1.7 times (measured on <NODE> in §8).

| # | Job | Command | Shards × items | Expected / shard (slow node) | Output | Disk |
|---|---|---|---|---|---|---|
| 1 | single-talker reverberation devel (cv_dev93) | `sbatch scripts/lora/reverb_single.sbatch devel 0 1` | 1 × 428 | < 1 minute (< 2) | `$SHARED/data/lora_v1/devel_rev_cv_dev93/` | 39 MB |
| 2 | single-talker reverberation train (train_si284; prerequisite for step-1 training and step-2 transcription) | `... train 0 1` | 1 × 3,513 | about 3 minutes (about 6) | `$SHARED/data/lora_v1/train_rev_si284/` | about 0.33 GB |
| 3 | P2-428 (devel clean) | `sbatch scripts/lora/baseline_audio.sbatch p2 devel_clean K 2` | 2 × 214 | about 5 minutes (about 9) | `outputs/lora_base_audio/p2_devel_clean/part{K}of2_<job>/P2/` | about 2 MB |
| 4 | Whisper + T2-428 (devel clean) | `... whisper_t2 devel_clean K 2` | 2 × 214 | about 6.5 minutes (about 11) | `.../whisper_t2_devel_clean/.../W/`, `T/` | about 2 MB |
| 5 | Whisper transcription of 7,026 train items (3,513 clean + 3,513 reverberant) | `... whisper train_all K 6` | 6 × 1,171 | about 7 minutes (about 12) | `.../whisper_train_all/.../W/` | about 3 MB |
| (6) | P2 / Whisper + T2 on C1-R (needed by plan §5, §8; not included in this request) | `... p2 devel_rev K 2`, `... whisper_t2 devel_rev K 2` | 2 × 214 each | same as 3, 4 | same as above | about 4 MB |
- Order: 1 and 2 (CPU) first; 3 and 4 do not depend on reverberation; 5 and 6 after 2 / 1 complete. GPU total about 1.1 h (excluding 6), about 1.5 h with 6. `<SHARED_STORAGE>` is now 88% used, with about 129 GB remaining.
- The segmentation and timing of step-1 training itself (7,026 audio items × 3 epochs, 2 settings) will be reported separately from measured speeds after the baselines above complete.

## 11. Supplementary reporting for the degradation checks (2026-11-06, decision of the project lead; **written before any step-1 results**; report only, not changing the §6 verdicts)
- On C5 (selected stream of the SpatialNet snapshot) and the c6-real-v1 main conditions (no abstention / abstention A / abstention B, the latter two on answered samples only), whenever the weighted interferer-only rate of a LoRA model (the audio LoRA, or the step-2 Whisper + T2-LoRA) is **higher than the corresponding baseline (any rise, not only above the threshold)**, also report:
  - on **stream-selection error samples** (using the existing selection results and verdicts, without changing the selection rule: the selected stream corresponds to the interferer), for the LoRA model and the baseline each: total predicted entities, mean per sample, how many of them match interferer gold values (D_I), and how many match target gold values (D_T);
  - the same numbers on correctly selected samples as a control.
- Purpose: distinguish "the model got worse" (extracting more interferer entities or fewer target entities even when selection is correct) from "after the model got stronger, the consequences of wrong selection are more visible" (extracting more entities only on selection-error samples). C3 has no stream selection; when its interferer-only rises, the same numbers are reported over all samples.

## 12. Step-0 report half and revision 1 item 4 (`eval_text_lora_gate.py report`, output `outputs/lora_text_gate/step0_report/report.json`)
- Report-half inference: job 22024104 (`eval_text_lora_ckpt.sbatch 1e-4 296 report`, 1 job, 1 checkpoint), <NODE>, A100-PCIE-40GB, 20:51:17–21:00:02 (8:45; inference 375 s), exit code 0, commit `774b621`, no uncommitted dependencies; 214 outputs; the sum of absolute values of lora_B of the loaded adapter is 4658.6767, the same as step 296 of selection-half job 22024022.
- Best checkpoint (1e-4 step 296) relative to the C0-428 baseline (job 22023517); bootstrap as in §5 (grouped by slurp_id, B = 2000, seed 2026; report half and all reported only):

| | Selection half (214 / 191) | **Report half (214 / 191)** | All (428 / 382) |
|---|---|---|---|
| At least one gold value hit: LoRA / baseline | 0.911 / 0.715 | **0.958 / 0.724** | 0.935 / 0.720 |
| Difference [95% interval] | +0.196 [0.133, 0.264] | **+0.234 [0.172, 0.295]** | +0.215 [0.171, 0.263] |
| 124 mix_v1 recordings by pair: LoRA / baseline (count) | 0.905 / 0.762 (63) | 0.951 / 0.803 (61) | **0.927 / 0.782** (124) |
| Difference on 124 [95% interval] | +0.143 [0.062, 0.237] | +0.148 [0.063, 0.254] | +0.145 [0.087, 0.212] |
| date recall: LoRA / baseline | 0.873 / 0.365 | 0.969 / 0.631 | 0.922 / 0.500 |
| Strict intent accuracy: LoRA / baseline | 0.907 / 0.668 | 0.911 / 0.743 | 0.909 / 0.706 |
| Parse failures: LoRA / baseline | 0 / 1 | 0 / 2 | 0 / 3 |
- c1-error-taxonomy-v1 class counts (LoRA; change relative to the baseline in parentheses):
  - selection half: a 0 (−1), b 1 (0), c 20 (−14), d 5 (−23), e 15 (−38), f 3 (0)
  - report half: a 0 (−2), b 0 (0), c 7 (−16), d 7 (−20), e 24 (−44), f 1 (0)
  - all: a 0 (−3), b 1 (0), c 27 (−30), d 12 (−43), e 39 (−82), f 4 (0)
- Per §4: "improvement" for external reporting is judged only on the report half; the lower bound of the report-half difference interval is 0.172 > 0. Step 0 only measures the gain from "learning SLURP's annotation spans" (without listening) and is not evidence that the audio LLM is better.
- Plan §6 item 5, "no C1 degradation for the text LoRA" (the text LoRA on the Whisper + T2 route), has not been run; whether to do it needs a separate decision.

## 13. Decision record (2026-11-06 21:1x EST, project lead) and submission order
1. Approved jobs 1–5 of §10 (devel reverberation, train reverberation, P2-428, Whisper + T2-428, Whisper transcription of the 7,026 train items), submitted with the configuration and dependencies of §10.
2. Approved the C1-R baselines (P2 and Whisper + T2 on reverberant devel, 2 shards each, configuration as in §10 items 3 and 4).
3. Approved running the text LoRA (1e-4 step 296) once on the Whisper transcripts of §10 item 4 (428 clean devel recordings), as the check "the text LoRA does not degrade on C1" (plan §6 item 5); **the result is labelled as a diagnostic, not as a step-2 result**. The control is the non-fine-tuned T2 on the same transcripts (the T output of item 4).
- All jobs have a 20-minute limit; after each job, check with sacct and the output files before submitting the jobs that depend on it; timeouts or failures are only reported, with no resubmission with changed configuration. Reruns after SpatialNet completes are scheduled before these jobs.
- Dependencies: job 1 → C1-R (P2, Whisper + T2 devel_rev); job 2 → job 5; job 4 → text LoRA diagnostic.

## 14. Baselines for steps 1 and 2 and the text LoRA diagnostic: run records and results (2026-11-06)
- All jobs exit code 0; all GPU jobs on A100-PCIE-40GB, commit `45fdfd8` (`fb55f64` for the reverberation jobs), no uncommitted dependencies, 20-minute limit, **no timeouts**.

| Job | Content | Node | Duration |
|---|---|---|---|
| 22024225 | devel reverberation (CPU) | <NODE> | 0:24 |
| 22024226 | train reverberation (CPU) | <NODE> | 2:42 |
| 22024227 / 22024228 | P2 devel clean (2 shards) | <NODE> / <NODE> | 6:47 / 5:02 |
| 22024229 / 22024230 | Whisper + T2 devel clean | <NODE> / <NODE> | 7:40 / 7:04 |
| 22024233 / 22024234 | P2 C1-R | <NODE> / <NODE> | 4:53 / 5:03 |
| 22024235 / 22024236 | Whisper + T2 C1-R | <NODE> / <NODE> | 5:37 / 5:40 |
| 22024240–22024245 | Whisper transcription of train (6 shards × 1,171) | <NODE> / 41 / 54 / 01 / 41 / 54 | 4:43–6:36 |
| 22024551 / 22024552 | **diagnostic**: text LoRA (1e-4 step 296) on Whisper transcripts | <NODE> / <NODE> | 9:28 / 8:30 |
- Reverberant audio check: devel 428 / train 3,513 files, 16-bit, 16 kHz, lengths matching the manifest; file sets matching the split_v2 lists (train has no intersection with devel / test); RIR samples all distinct (cv_dev93 428, train_si284 3,513). **1 train recording is clipped**: `audio-1497882525-headset.flac` (train_si284 sample 27624, T60 0.201 s), peak 1.305 after level adjustment, 8 samples truncated when writing 16-bit (the dry-run covered only the first half and did not catch it); not regenerated. No clipping in devel. Disk: devel 38 MB, train 322 MB.
- CPU dry-run: text LoRA diagnostic srun 22024550. Evaluation: `scripts/lora/eval_baselines428.py` (read-only on the login node), results `outputs/lora_base_audio_eval/v1/eval_baselines428.json`.
- **Consistency**: identical item by item with the existing 124-item runs — P2 (22014089 C1) 124 / 124, W_C1 and T_C1 (22016221) 124 / 124 each. The input transcripts of T are identical to the W outputs 428 / 428; the input prompts of the diagnostic are identical to T 428 / 428. 7,026 train transcripts (clean 3,513 + reverberant 3,513), no empty transcripts.
- Results ("at least one gold value hit", single-talker basis; selection half / report half / all):

| Condition | At least one hit | date recall | Strict intent accuracy | Parse failures (all) | Whisper WER (all) |
|---|---|---|---|---|---|
| P2 (clean) | 0.636 / 0.603 / **0.619** | 0.317 / 0.508 / 0.414 | 0.687 / 0.738 / 0.713 | 0 | — |
| Whisper + T2 (clean) | 0.607 / 0.575 / **0.591** | 0.317 / 0.508 / 0.414 | 0.664 / 0.701 / 0.682 | 2 | 0.067 |
| P2 (C1-R) | 0.579 / 0.607 / **0.593** | 0.317 / 0.508 / 0.414 | 0.650 / 0.701 / 0.675 | 0 | — |
| Whisper + T2 (C1-R) | 0.523 / 0.481 / **0.502** | 0.286 / 0.446 / 0.367 | 0.626 / 0.626 / 0.626 | 3 | 0.142 |
| C0-428 (reference transcripts + T2, §2) | 0.715 / 0.724 / 0.720 | 0.365 / 0.631 / 0.500 | 0.668 / 0.743 / 0.706 | 3 | — |
| **Diagnostic**: text LoRA on Whisper (clean) | 0.822 / 0.841 / 0.832 | 0.841 / 0.938 / 0.891 | 0.869 / 0.869 / 0.869 | 0 | (same as above, 0.067) |
- The identical date recall of the first three rows is a coincidence: each hits 53 / 128 date entities, but different ones (pairwise overlap 47).
- Error classes (all 428): P2 a 0, b 1, c 55, d 58, e 180, f 2; Whisper + T2 a 2, b 1, c 53, d 57, e 190, f 3; P2-R a 0, b 1, c 51, d 48, e 205, f 3; Whisper + T2-R a 3, b 1, c 52, d 55, e 232, f 3; diagnostic a 0, b 1, c 35, d 17, e 89, f 3.
- **Diagnostic (plan §6 item 5, "no C1 degradation for the text LoRA", against the non-fine-tuned T2 on the same transcripts)**: selection half +0.215 (intent +0.206, parse failures 0), report half +0.266, all +0.241 (bootstrap reported only: all [0.191, 0.289]); **no degradation** under the point-estimate thresholds of §6. Diagnostic only, **not a step-2 result** (the step-2 Whisper + T2-LoRA must be trained on Whisper transcripts).

## 15. Rule supplements for steps 1 and 2 (2026-11-06 22:0x EST, decision of the project lead; **written before any step-1 training or results**)
1. The step-1 audio LoRA (L1) configuration is approved as proposed after §14: train 3,513 clean + 3,513 train_si284 reverberant = 7,026 items; audio + frozen P2 → JSON in the same format as step 0 (transcript = original SLURP text), loss on the answer tokens only (entity-part loss observed only); LoRA and optimisation settings as in step 0; learning rates {5e-5, 1e-4}; micro-batch 4 × accumulation 4 = 16; 440 steps per epoch (the last step with 2 items), 1,320 steps over 3 epochs, warmup round(0.03 × 1320) = 40; checkpoints at steps 220, 440, 660, 880, 1100, 1320; seed 2026, shuffled each epoch with `random.Random(2026 + epoch)`. First write the script, CPU dry-run and commit, then submit the pilot test (1 job), report from measurements, and start formal training after approval.
2. **Training segments**: Slurm limit 30 minutes, 27-minute guard inside the script; each segment starts with the same resume check as step 0 (exact-check failure → exit code 3, no training, stop and report). **Inference jobs**: limit 20 minutes, 1 checkpoint per job.
3. **Checkpoint selection (same for steps 1 and 2)**: only "at least one gold value hit" on the **C1 selection half** (214 recordings / 191 sentences) is considered (single-talker basis, as in step 0); highest point estimate, ties to fewer steps, then the smaller learning rate. C1-R is reported only. The step-2 (Whisper + T2-LoRA) checkpoint is selected on the Whisper transcripts of the C1 selection half, with exactly the same rule.
4. **Supplement to the primary comparison**: besides C1, the primary comparison of steps 1 and 2 (audio LoRA − Whisper + T2-LoRA, paired difference on the same audio) **also reports** the paired difference and 95% interval **on C1-R** (same bootstrap: grouped by slurp_id, B = 2000, `np.random.default_rng(2026)`; selection half, report half and all reported separately; external judgement still on the report half only).
5. The 1 clipped reverberant train recording (`audio-1497882525-headset.flac`, peak 1.305, 8 samples truncated, see §14) is **not regenerated** and is used as usual for training and step-2 transcription.
6. **GPU budget**: as of §14 about 6.1 GPU hours have been used (33 jobs). If at any point the projected total would exceed 20 hours, stop and report first; **do not reduce checkpoints or hyperparameter settings on one's own initiative**.

## 16. Step-1 training script and pilot test (implementation conventions, **written before any step-1 training**)
- Script: `scripts/lora/train_audio_lora.py` / `.sbatch` (30 minutes; `lr_at` and `target_json` of step 0 imported directly). Input = the same input as frozen P2 inference (chat-template text + processor-expanded audio placeholders, right padding); the prompt tokens of each item are identical token by token to single-item inference `build(qp, PROMPT_P2, wav)` (dry-run assertion). Target and loss as in step 0. Clean audio is input as is (same as C1 inference, no level adjustment); reverberant audio is read from `data/lora_v1/train_rev_si284/`. Script assertions: the intersection of sentences, slurp_ids, speakers and files between train and devel / test is 0; the reverberant manifest matches the train list and all RIRs come from train_si284; trainable parameters exist only in the 112 language-model modules, and the audio encoder has no trainable parameters.
- Segments: each segment starts from the previous resume point and runs until step 1,320 or until elapsed time exceeds **24 minutes** (checked between optimizer steps); if not finished, it saves the resume state (adapter, optimizer, RNG, global step, checksum) to `resume_step{S}/` and then runs 2 reference steps; 27-minute hard guard (exit code 4). The next segment reads `state.json`, performs the same exact checks as step 0 (failure → exit code 3, no training, stop and report), then reruns the 2 reference steps and compares (report only). Segment boundaries are determined by elapsed time, so the number of steps per segment is not fixed; the log records the start and end step of each segment. Checkpoints at steps 220, 440, 660, 880, 1100, 1320 (if the reference steps cross a checkpoint, it is saved in the next segment).
- CPU dry-run: srun 22024590 (<CPU_PARTITION>, <NODE>): planned 1,320 steps, warmup 40, checkpoints 220–1320; the 8 prompts of the first 2 micro-batches are 602–682 tokens, targets 34–42 tokens; prompts identical to single-item inference inputs, target decoding identical to the gold JSON; `DRY_RUN_PASS`.
- Pilot test (1 job, 30 minutes): `sbatch scripts/lora/train_audio_lora.sbatch 5e-5 1 pilot 40`, training 40 optimizer steps from step 0 on the formal schedule, then saving the resume state and running 2 reference steps, written to `checkpoints/lora_v2/step1_pilot/<作业号>/` (measures speed and GPU memory only; not used for formal training).
- **Pilot test result**: job 22024591, <NODE>, A100-PCIE-40GB, 22:12:15–22:17:42 (5:27), exit code 0, MaxRSS 26.0 GiB, commit `b5b4284`, no uncommitted dependencies. Loading 84.5 s (GPU memory after loading 16.68 GiB). 40 optimizer steps (micro-batch 4 × 4): median **5.11 s** per step (4.90–5.48; step 1 9.4 s including warm-up), i.e. **0.319 s** per item; peak GPU memory allocated 20.83 GiB (the same as other jobs, possibly from the loading peak; far below the 38 GB stopping line). Saving the resume state 10.9 s (adapter 39 MB, `resume.pt` 78 MB). Loss (observation only): mean of the first 10 steps 0.248, last 10 steps 0.120; entity part 0.265 → 0.133.
- Re-estimate from measurements: 1,320 steps × 5.11 s ≈ 1.87 h of training per setting; each segment about 24 minutes minus about 1.5 minutes of loading, about 265 steps / segment → about 5–6 segments per setting (about 9 on slow nodes), plus about 2 minutes of overhead per segment, about 2.1 h per setting, about 4.2 h for two settings.

## 17. Decision record: step-1 formal training (2026-11-06 22:2x EST, project lead)
- Approved step-1 formal training: segment 1 of both settings (5e-5, 1e-4) submitted at the same time; after each segment, check with sacct and the log before submitting the next segment of the same setting; each segment starts with the resume check, and on failure stop and report.
- **The total LoRA GPU budget cap is raised from 20 hours to 24 hours** (the 20 hours of plan §8 and §15 item 6 of this file; SpatialNet reruns not counted). If the projection exceeds 24 hours, stop and report; still no self-initiated reduction of checkpoints or hyperparameter settings.
- GPU memory: from segment 1 of formal training on, record the loading peak after loading the model, then `torch.cuda.reset_peak_memory_stats()`; the `SEGMENT_DONE` line reports the allocated / reserved peaks of the training phase (the peak of the `STEP` lines is also for the training phase). The 20.83 GiB of the pilot test includes the loading peak.
- After training: run the 12 checkpoints on the C1 selection half (1 per job, 20 minutes), select the best per §15 item 3, apply the gate per plan §7 item 5 (audio LoRA − P2(C1), selection-half paired bootstrap lower bound > 0), and then report. Also report, on the step-0 basis, the mean training loss of the first 0.25 epoch (steps 1–110) and the last 0.25 epoch (steps 1211–1320) (report only).
- Step-1 checkpoint inference script: `scripts/lora/eval_audio_lora_ckpt.py` / `.sbatch` (20 minutes, 1 checkpoint per job; the same audio, `build()` and decoding as `baseline_audio.py --mode p2`, only additionally loading the LoRA adapter). CPU dry-run: the srun above (C1 selection half 214, report half 214, all C1-R 428; inputs constructed successfully).
- Two generalisations of the gate script `eval_text_lora_gate.py` (**logic unchanged**): `--baseline` accepts several raw.jsonl joined by commas (the P2 baseline has 2 shards); the loss windows take the first / last 0.25 epoch according to `--total-steps` (444 → steps 1–37 / 408–444, 1320 → steps 1–110 / 1211–1320; the output field names remain `first37_mean` / `last37_mean`). Check: rerunning with the 18 step-0 candidates gives `best`, `bootstrap`, `exceptions`, `verdict` and `candidates` exactly identical to `outputs/lora_text_gate/step0_selection/gate.json`; the P2 baseline (2 shards) compared with itself gives difference 0 and interval [0, 0]. Step 1 is judged per plan §7 item 5 (lower bound ≤ 0 → stop, no L2, step 2 still done); the step-0 exception items in the script are only reported for step 1.

## 18. Step-1 training and gate results (2026-11-07)
- Training (`train_audio_lora.sbatch`, 30-minute limit, commit `9cab32d`, no uncommitted dependencies, A100-PCIE-40GB; every segment exit code 0; the 27-minute guard was never triggered):

| Setting | Segments (job, node, steps) |
|---|---|
| 5e-5 | 1 (22025105, <NODE>, 1–271); 2 (22025138, <NODE>, 272–531); 3 (22025177, <NODE>, 532–790); 4 (22025203, <NODE>, 791–1074); 5 (22025231, <NODE>, 1075–1320) |
| 1e-4 | 1 (22025106, <NODE>, 1–287); 2 (22025144, <NODE>, 288–557); 3 (22025183, <NODE>, 558–827); 4 (22025205, <NODE>, 828–1118); 5 (22025248, <NODE>, 1119–1320) |
- Median per step 4.86–5.14 s; loading 23–102 s (about 100 s when the shared file system was slow); training-phase peak GPU memory allocated 19.58 GiB / reserved 21.03 GiB (loading peak 20.83 GiB). Each setting's `log.jsonl` has 1,320 steps, contiguous, no duplicates, no non-finite loss; all 6 checkpoints present.
- **Resume checks**: all 8 exact checks (segments 2–5 of each setting) pass. Reference-step comparison (report only): mostly relative difference 0; 5e-5 segment 2 step 2 checksum 5.3e-9; segment 4 step 1 checksum 2.6e-8, step 2 loss 1.6%, checksum 2.9e-7; segment 5 step 2 checksum 1.2e-10 (tiny non-determinism of GPU computation).
- Training loss (total / entity part, report only): 5e-5 steps 1–110 0.134 / 0.137, steps 1211–1320 0.023 / 0.012; 1e-4 0.119 / 0.120 → 0.012 / 0.005.
- Checkpoint inference (`eval_audio_lora_ckpt.sbatch`, C1 selection half, 1 per job, 20 minutes, commit `9cab32d`): 22025268–22025273 (5e-5), 22025279–22025284 (1e-4), all exit code 0, 7:33–13:10, 214 items each.
- Gate (`eval_text_lora_gate.py gate --total-steps 1320`, output `outputs/lora_audio_gate/step1_selection/gate.json`); C1 selection half "at least one gold value hit", P2 baseline 0.636:

| Setting \ step | 220 | 440 | 660 | 880 | 1100 | 1320 |
|---|---|---|---|---|---|---|
| 5e-5 | 0.832 | 0.827 | 0.822 | 0.860 | 0.860 | 0.855 |
| 1e-4 | 0.846 | **0.860** | 0.841 | 0.846 | 0.841 | 0.855 |
- **Best: 1e-4 step 440** (0.860; tied with 5e-5 steps 880 and 1100; fewer steps by the rule). Paired difference +0.224, 95% interval **[0.156, 0.295]**, lower bound > 0 (the first part of plan §7 item 5 passes; the best is selected on the same selection half, so the lower bound is optimistic). Parse failures 0 / 214.
- Best checkpoint on the C1 selection half (descriptive only): 240 / 298 gold entities hit; date recall 0.889 (P2 0.317); strict intent accuracy 0.897 (P2 0.687); classes a 0, b 1, c 23, d 8, e 25, f 1; zero entities 0, abstentions 0. The point-estimate threshold of §6 item 1 (C1) is met; the other items of §6 (C4-direct, c6-real-v1, C3, C5) have not yet been run.

## 19. How the SpatialNet completion went unreported (recorded 2026-11-07)
- SpatialNet-small training job 22010857 ended at **2026-11-06 23:23:10** (COMPLETED, exit code 0, run time 1-15:20:14, <NODE>). The working rules require reporting first and waiting for approval when it completes, and its reruns take priority over LoRA.
- I last confirmed it was running at 22:20 (when reporting the §16 pilot test). After that only the end states of the LoRA jobs were monitored, and **22010857 was not checked again** until it was noticed at 11-07 02:1x while tallying GPU usage, about 3 hours late.
- In the meantime, segments 2–5 of step-1 LoRA training (8 jobs) and 12 checkpoint inference jobs were submitted. No SpatialNet rerun job was queued in that period, so no GPU was actually taken from it, but the rule "report first and wait for approval on completion" was violated.
- Improvement (decision of the project lead 2026-11-07): from now on, before every report, check the status of all relevant jobs (including non-LoRA jobs) with sacct.

## 20. Revision: remaining step-1 checks (decision of the project lead 2026-11-07; **written before any results of these checks**)
1. Handle SpatialNet first (read-only search for the final checkpoint, list the job configurations of each rerun step, wait for approval); all LoRA jobs are paused. The remaining step-1 checks come after the SpatialNet rerun; step 2 (Whisper + T2-LoRA) starts after the step-1 degradation checks are complete.
2. The conditions of plan §6 that involve SpatialNet (C5, c6-real-v1) **switch to the separation results of the final checkpoint**, with the baseline being P2 on the same separated speech (from the SpatialNet rerun); **the ep72 snapshot is no longer used**.
3. The degradation verdicts of the mixture conditions (C4-direct, C5, c6-real-v1, C3) use **all 200 samples**; reason: the checkpoint has already been selected on C1 only, and the mixture conditions take no part in selection. The selection half and report half (group split of c6-delta-v1) are reported separately. The supplementary reporting of §11 applies as usual.
4. Jobs that do not depend on SpatialNet (C4-direct, C3, C1 report half, C1-R) may be submitted after the SpatialNet rerun jobs are submitted; the scripts are written first and CPU dry-run.

## 19b. Correction (2026-11-07 10:3x EST, after a read-only check): SpatialNet **training had not completed**
- The COMPLETED of 22010857 was only **the end of one continuation segment**: the log `speech_separation/multichannel/logs/spatialnet_train_22010857.out` continues training from epoch 65 and ends at epoch 87, with the last line `CONTINUATION_REQUIRED`; `train.sbatch` automatically submits the next segment at the end (`continuation_jobs.txt` records 22025125). Continuation job **22025125** started at 23:23:20, RUNNING (<NODE>, limit 2 days), and by 10:33 had completed up to epoch 93. The total number of epochs in `run_config.json` is 100.
- The judgement in §19, "training completed and went unreported", therefore does not hold: the event of training completion had not yet happened. The remaining problem is that at 02:1x I concluded from the sacct state of 22010857 alone, without checking the log and the continuation job, and the report was wrong. The improvement at the end of §19 (check all relevant jobs with sacct before reporting) applies as usual, with the addition: whether training is complete must be judged from the log and continuation records, not only from the job state.
- The revision of §20 remains in effect (C5 / c6-real-v1 switch to the **final** checkpoint, i.e. the checkpoint obtained after epoch 100 by the selection rule of `run_config.json`).

## 21. Advance revision (decision of the project lead 2026-11-07; **written before any results of the SpatialNet final evaluation and the remaining step-1 checks**; no frozen experimental rule is changed)
1. Approved the configuration of steps 0–6 of the SpatialNet final rerun (see the 11-07 10:35 report: 0 confirm completion → 1 read-only copy of the final best.pt → 2 cv_dev93 check + mix_v1 separation → 3 mix_v1 stream selection → 4 P2 / Whisper + T2 on the selected stream → 5 P2 / Whisper + T2 on the other stream → 6 recompute C5, C6δ, c6-real-v1 under the frozen rules). **After training completes and passes the established completion criteria** (`frontend_compare_v1.md`: the run directory has `test_summary.json`, checkpoint `complete == True`, the training job that produced it has ended; additionally check that the log reaches epoch 100 and there is no new continuation job), **report first, then start step 1**.
2. **Steps 4 and 5 keep the 30-minute limit** (the configuration of the original scripts `qwen_omni_snapshot*.sbatch`), recorded as a specific exception for this final evaluation; other new jobs remain at 20 minutes.
3. The order of §20 item 4 is changed: the four step-1 checks that do not depend on SpatialNet (inference of the best audio LoRA checkpoint 1e-4 step 440 on **C4-direct, C3, the C1 report half, C1-R**) **may be run early while waiting for SpatialNet training**; the scripts are written first and CPU dry-run. The SpatialNet final evaluation still has the highest priority (when both are waiting for GPUs, the SpatialNet jobs are submitted first).
4. The verdicts of these four items follow plan §6 and §20 without changing thresholds: C4-direct target-any not below P2 − 0.03 (all 200 samples); C3 interferer-only no more than 0.02 above P2 and target-any not below P2 − 0.03 (all 200); selection half / report half (c6-delta-v1 group split) reported separately; on any rise in interferer-only, entity counts are reported per §11. The C1 report half and C1-R are reported only (§6 item 1 for C1 has already been judged on the selection half). Baselines are the existing P2 outputs (C4-direct, C3: P2 outputs of the original jobs, same code path; C1, C1-R: the 428-item baselines of §14).
5. test is not used, other running training jobs are not affected, and no new LoRA training is started; failures or anomalies are reported first, with no self-initiated retry.

## 22. Correction of the SpatialNet completion criteria (2026-11-07, a read-only check relayed by the project lead; **written before the final model was copied and evaluated**; no original checkpoint is modified)
- Problem: SpatialNet's `best.pt` is updated only when the validation loss improves, so after training completes it may still be `complete=False`.
- Corrected completion criteria (all must be met):
  1. **`latest.pt`**: `complete == True`, and `epoch == 100`;
  2. **`best.pt`**: `complete == True` is **not required**; its epoch must equal the epoch with the minimum validation loss in the complete `metrics.jsonl` (100 entries) (`run_config.json`: selection = minimum full-dev PIT negative SI-SDR), with the same validation loss value;
  3. the rest unchanged: the continuation job chain has ended (no new continuation job running or queued), the log completes epoch 100, the run directory has `test_summary.json`, and the checkpoint configuration agrees with `run_config.json` (equal after a JSON round trip).
- The final evaluation uses the `best.pt` confirmed by criterion 2 (the same selection rule as for the snapshot); copying is read-only and the original files are untouched.
- Scripts for the four checks that do not depend on SpatialNet: `scripts/lora/eval_audio_lora_mix.py` / `.sbatch` (C4-direct, C3; inputs constructed by functions of the original frozen P2 scripts, imported unchanged) and the existing `eval_audio_lora_ckpt.py` (C1 report half, C1-R). CPU dry-run (<CPU_PARTITION> srun): 200 items each for C4-direct / C3, the prompt texts of the first 5 identical character by character to `prompt_text` in the baseline raw outputs; inputs for the C1 report half (214) and C1-R (428) constructed successfully.

## 23. Remaining step-1 checks (after the change of executor) (2026-11-07)
- The project lead arranged a change of executor, who took over the subsequent experiments, status checks and monitoring. The frozen rules and approved configurations of §20–22 are kept; Whisper + T2-LoRA is not started, the final test is not used, and other training is not interfered with.
- The audio LoRA is fixed at **1e-4 step 440**; even if the C3 degradation check fails, no checkpoint is reselected, no threshold adjusted, and no existing inference rerun.
- Five inference jobs submitted: 22027859 (C4-direct), 22027860 (C3), 22027861 (C1 report half), 22027862 (C1-R selection half), 22027863 (C1-R report half), all with a 20-minute limit. Results are computed by reusing `eval_step1_checks.py`, without changing its main scoring, thresholds or bootstrap.
- At handover, HEAD was `4393e63`, the working tree clean and consistent with the local origin/master; these jobs were not resubmitted.

### Post-hoc C3 diagnosis (descriptive; takes no part in any selection or gate)
- After the C3 degradation result was known, the project lead asked for an additional class and entity-count analysis. The script `scripts/lora/diagnose_c3_binding.py` imports the existing scoring functions directly; output `outputs/lora_step1_checks/c3_diagnostic_20261107/{diagnostic.json,paired_samples.jsonl}`, which saves the SHA256 of both inputs and the per-sample paired classification without overwriting the original results.
- **Note on basis**: the classification of the frozen binding-score-v1 is based on "all distinguishable target values hit" and "at least one distinguishable interferer value hit". `neither` may therefore include partial target hits, and `only_interferer` may in general also include partial target hits. The table below keeps the original definition; a diagnostic basis with "at least one hit" for both sides is reported in addition.

| Frozen four classes (200 items) | P2 count / proportion | Audio LoRA count / proportion |
|---|---|---|
| only_target | 21 / 0.105 | 33 / 0.165 |
| only_interferer | 23 / 0.115 | 44 / 0.220 |
| both | 0 / 0 | 0 / 0 |
| neither | 156 / 0.780 | 123 / 0.615 |

| Supplementary diagnosis: at least one distinguishable value hit for each side | P2 count | Audio LoRA count |
|---|---|---|
| hit for target only | 25 | 36 |
| hit for interferer only | 23 | 44 |
| hits for both | 0 | 0 |
| no distinguishable value hit for either side | 152 | 120 |

- Parse failures are 0 for both routes. Among "no distinguishable value hit for either side", P2 has 10 items with empty entities and 142 that output entities but hit no distinguishable value; LoRA 0 / 120. Hitting no distinguishable value does not mean having no content at all or all entities being wrong; values shared by both sides are not in the distinguishable-value sets.
- Total entities P2 201 → LoRA 253; per sample 1.005 → 1.265; interferer distinguishable-value hits 23 → 47; target distinguishable-value hits 25 → 40.
- Paired transitions (main basis, row P2 → column LoRA): only_target → only_target 17, only_interferer 2, neither 2; only_interferer → only_interferer 21, neither 2; neither → only_target 16, only_interferer 21, neither 119; all cells involving both are 0.
- Of the 23 new only_interferer items, 21 were previously neither and 2 previously only_target; another 2 previously only_interferer became neither, a net increase of 21. **This confirms that interferer distinguishable-value hits increased in the outputs, but the existing outputs cannot causally distinguish "stronger extraction" from "more speaker-binding errors".** C5 / c6-real-v1 on the same separated speech from the final SpatialNet, and the §11 diagnostics grouped by routing correctness, are still to be run.
- **The C3 main verdict remains fail**: only_interferer increases by 0.105, above +0.02; target_any 0.125 → 0.180 meets the other requirement of not being below P2 − 0.03. C4-direct target_any 0.670 → 0.875 passes its degradation check. C3 already means §6 cannot be judged as passing overall; the remaining items continue to be completed and reported; unfinished items are not marked as passing.

### Other training status (read-only check, no intervention)
- SpatialNet 22025125 has completed epoch 95/100, best dev SI-SDR 20.240656956 dB (ep95, valid_loss −20.240651719); still running, no new continuation, no test_summary.json yet. The completion check continues per §22; the final model has not been copied and the final evaluation not started.
- MossFormer2 21972915: sacct shows TIMEOUT; the log clearly records termination at 2026-11-07 10:55:52 due to TIME LIMIT. Kept as a separate to-do item; not resubmitted.

### Formal summary after the five jobs completed (2026-11-07)
- All five jobs COMPLETED / ExitCode 0:0, every log has CKPT_DONE and ALL_DONE; durations 8:02, 7:12, 7:48, 10:04, 7:49 (40:55 of GPU time in total). 22027862 ran on <NODE>, the rest on <NODE>, all A100-PCIE-40GB.
- C4-direct / C3 200 items each; the C1 report half and both halves of C1-R 214 items each. Checks of the sample ID sets, duplicates, frozen splits and prompt consistency pass; for the single-talker conditions it was additionally checked that audio paths, slurp_id, gold transcripts, intent and entities are exactly identical to the respective baselines. The configs of the three single-talker jobs are all step440, with the correct condition/half and the expected job IDs; the sum of absolute values of the adapter lora_B is 6594.5746 in all of them.
- The inference code version is 4393e63. The run provenance of 22027862 recorded the untracked file scripts/lora/diagnose_c3_binding.py (a post-hoc analysis script added while waiting for the jobs); this file is not imported by inference, and no inference dependency was modified. The scripts/lora and scripts/qwen working-tree status recorded by the other two single-talker jobs is empty.
- Full evaluation: `envs/qwen_omni_lora/bin/python -I -B scripts/lora/eval_step1_checks.py outputs/lora_step1_checks/nonspatial_20261107`; results `eval_step1_checks.json` and input audit `input_audit.json` (raw-output SHA256, job configs, input-consistency assertions), both in that directory.
- The single-talker paired bootstrap directly reuses the existing boot function: grouped by slurp_id, recording-weighted difference, B=2000, seed=2026. All differences in the tables below are LoRA − P2; external improvement is judged only on the report half; the selection-half intervals are optimistic because of checkpoint selection.

| Condition | Half | P2 | Audio LoRA | Difference [95% CI] |
|---|---|---|---|---|
| C1 | selection | 0.636 | 0.860 | +0.224 [0.156, 0.295] |
| C1 | report | 0.603 | 0.893 | +0.290 [0.217, 0.367] |
| C1 | all | 0.619 | 0.876 | +0.257 [0.206, 0.310] |
| C1R | selection | 0.579 | 0.790 | +0.210 [0.147, 0.280] |
| C1R | report | 0.607 | 0.836 | +0.229 [0.159, 0.299] |
| C1R | all | 0.593 | 0.813 | +0.220 [0.172, 0.269] |

| Condition | Half | date recall P2 → LoRA | Intent accuracy P2 → LoRA |
|---|---|---|---|
| C1 | selection | 0.317 → 0.889 | 0.687 → 0.897 |
| C1 | report | 0.508 → 0.923 | 0.738 → 0.864 |
| C1 | all | 0.414 → 0.906 | 0.713 → 0.881 |
| C1R | selection | 0.317 → 0.841 | 0.650 → 0.864 |
| C1R | report | 0.508 → 0.908 | 0.701 → 0.850 |
| C1R | all | 0.414 → 0.875 | 0.675 → 0.857 |

- LoRA parse failures are 0 in every half of C1 and C1-R. The lower bounds of the primary-metric difference intervals on the report halves of C1 and C1-R are both >0, showing an improvement over the non-fine-tuned P2; **this is not evidence that the audio LoRA outperforms Whisper + T2-LoRA**, which has not yet been trained.
- The formal C1 degradation gate still uses the selection half only: the primary metric, intent and parse failures all pass; the C1 report half and C1-R are reported only, with no new gate.

| Condition | Half | target-any P2 → LoRA | Interferer-only P2 → LoRA | Primary-metric difference [95% CI] |
|---|---|---|---|---|
| c4direct | selection | 0.620 → 0.840 | 0.000 → 0.000 | +0.220 [0.130, 0.320] |
| c4direct | report | 0.720 → 0.910 | 0.000 → 0.000 | +0.190 [0.120, 0.260] |
| c4direct | all | 0.670 → 0.875 | 0.000 → 0.000 | +0.205 [0.145, 0.270] |
| c3 | selection | 0.100 → 0.170 | 0.110 → 0.190 | +0.070 [0.020, 0.130] |
| c3 | report | 0.150 → 0.190 | 0.120 → 0.250 | +0.040 [-0.020, 0.100] |
| c3 | all | 0.125 → 0.180 | 0.115 → 0.220 | +0.055 [0.015, 0.095] |

- The mixture-condition intervals were filled in by `scripts/lora/bootstrap_step1_mix.py` with the paired group-level bootstrap already specified in §4 of the frozen plan: stratified by angle, the two swapped samples of a group drawn together, B=2000, seed=2026, equal-weight sample basis; output `outputs/lora_step1_checks/mix_bootstrap_20261107/bootstrap.json`. The original point-estimate gate is not changed. C3 interferer-only difference intervals: selection half +0.080 [0.040,0.130]; report half +0.130 [0.060,0.210]; all +0.105 [0.060,0.155]. For C4-direct the corresponding differences and intervals are all 0.
- **§6 status**: C1 passes; C4-direct passes; C3 fails (original verdict unchanged); the final-SpatialNet C5 pair-flip accuracy and the c6-real-v1 direction-risk checks await the final front end and have not yet been run. The overall §6 failure is already determined by C3; the C1/C1-R gains are not interpreted as all gates passing; L2 is not started, and Whisper + T2-LoRA remains on hold as requested by the project lead.
- This round had no new GPU jobs, no retries, no checkpoint changes and no use of test data. The subsequent final SpatialNet evaluation and the §11 analysis split by correct/incorrect routing continue under the approved plan.

## 24. SpatialNet final training completed (2026-11-07 22:33 EST)
- All §22 checks pass: the continuation chain ended normally, complete logs for epochs 1–100, latest epoch100/complete=True, best corresponds to the minimum validation loss over the complete history, configuration consistent with the source of the existing wrap-up summary. Best ep100, dev SI-SDR 20.31431 dB.
- Reported to the project lead first; then the final best.pt was copied and verified under the existing authorisation; the record and the subsequent final-evaluation configuration are in `docs/spatialnet_final_v1.md`. No new LoRA training in this round; the C3 fail verdict is not changed.

## 25. Audio LoRA checks on the final SpatialNet
- Fixed step1/lr1e-4/ckpt_step440; final SpatialNet ep100; the input audio SHA256 of both streams, the P2 prompt, sample labels and streams are identical item by item to the same-stream baseline. All gates use all 200 samples; no checkpoint is reselected.
- Results and raw-prediction SHA256: `outputs/lora_step1_checks/final_spatialnet_ep100_20261107/checks.json`. The consistency assertions against the original scoring function and the C6 primary metric pass; intervals are paired group bootstrap with B=2000, seed2026, stratified by angle.
- Baseline predictions: `outputs/qwen_omni_final/22029639_selected/`, `22029640_other/`; LoRA predictions: `outputs/lora_audio_final/22029676_selected/`, `22029677_other/`.

### C5 (equal-weight samples)

| Half | target-any P2→LoRA | Paired difference [95% CI] | Interferer-only P2→LoRA | Pair flip P2→LoRA | Other-stream interferer-only (report only) P2→LoRA |
|---|---:|---:|---:|---:|---:|
| selection | 0.5600 → 0.7900 | +0.2300 [0.1400, 0.3200] | 0.0000 → 0.0100 | 0.2400 → 0.6000 | 0.5700 → 0.8000 |
| report | 0.6800 → 0.8800 | +0.2000 [0.1100, 0.2900] | 0.0000 → 0.0000 | 0.3600 → 0.6600 | 0.7000 → 0.8800 |
| all | 0.6200 → 0.8350 | +0.2150 [0.1500, 0.2750] | 0.0000 → 0.0050 | 0.3000 → 0.6300 | 0.6350 → 0.8400 |

### Real C6 main conditions (natural angle weighting)
- With abstention, metrics use answered samples as the denominator; target hits keep the sector-indistinguishable penalty. For noise, proportions are first computed per repetition and then averaged over the 10 repetitions; the bootstrap draws the two swaps of a group and all noise repetitions together.

| Condition/abstention | Half | Target hit P2→LoRA | Target-hit difference [95% CI] | Interferer-only P2→LoRA | Interferer difference [95% CI] | Coverage | Main stream-selection error rate |
|---|---|---:|---:|---:|---:|---:|---:|
| sector_K8_center/none | selection | 0.4861 → 0.6819 | +0.1958 [0.0766, 0.3214] | 0.0350 → 0.0503 | +0.0153 [0.0000, 0.0332] | 1.0000 | 0.1643 |
| sector_K8_center/none | report | 0.5590 → 0.7707 | +0.2117 [0.0971, 0.3357] | 0.0639 → 0.0662 | +0.0023 [-0.0080, 0.0118] | 1.0000 | 0.1088 |
| sector_K8_center/none | all | 0.5252 → 0.7295 | +0.2043 [0.1209, 0.2907] | 0.0505 → 0.0588 | +0.0083 [-0.0004, 0.0189] | 1.0000 | 0.1345 |
| sector_K8_center/A | selection | 0.5698 → 0.8119 | +0.2422 [0.0933, 0.4002] | 0.0052 → 0.0104 | +0.0052 [0.0000, 0.0160] | 0.8084 | 0.0104 |
| sector_K8_center/A | report | 0.6161 → 0.8504 | +0.2343 [0.1066, 0.3732] | 0.0124 → 0.0083 | -0.0041 [-0.0127, 0.0000] | 0.8785 | 0.0124 |
| sector_K8_center/A | all | 0.5956 → 0.8334 | +0.2378 [0.1389, 0.3410] | 0.0092 → 0.0092 | +0.0000 [-0.0067, 0.0069] | 0.8460 | 0.0116 |
| sector_K8_center/B | selection | 0.5825 → 0.8192 | +0.2366 [0.0928, 0.3921] | 0.0000 → 0.0000 | +0.0000 [0.0000, 0.0000] | 0.8273 | 0.0000 |
| sector_K8_center/B | report | 0.6272 → 0.8648 | +0.2376 [0.1108, 0.3757] | 0.0000 → 0.0000 | +0.0000 [0.0000, 0.0000] | 0.8912 | 0.0000 |
| sector_K8_center/B | all | 0.6073 → 0.8445 | +0.2372 [0.1415, 0.3373] | 0.0000 → 0.0000 | +0.0000 [0.0000, 0.0000] | 0.8615 | 0.0000 |
| noise_sigma10/none | selection | 0.5302 → 0.7588 | +0.2286 [0.1106, 0.3544] | 0.0358 → 0.0487 | +0.0129 [0.0055, 0.0219] | 1.0000 | 0.0635 |
| noise_sigma10/none | report | 0.6136 → 0.8335 | +0.2199 [0.1070, 0.3463] | 0.0278 → 0.0324 | +0.0046 [-0.0010, 0.0099] | 1.0000 | 0.0361 |
| noise_sigma10/none | all | 0.5749 → 0.7989 | +0.2240 [0.1407, 0.3119] | 0.0315 → 0.0399 | +0.0084 [0.0040, 0.0137] | 1.0000 | 0.0488 |
| noise_sigma10/A | selection | 0.5664 → 0.8062 | +0.2398 [0.0937, 0.3916] | 0.0026 → 0.0046 | +0.0020 [0.0000, 0.0055] | 0.8186 | 0.0056 |
| noise_sigma10/A | report | 0.6247 → 0.8588 | +0.2342 [0.1061, 0.3759] | 0.0029 → 0.0025 | -0.0004 [-0.0025, 0.0012] | 0.8866 | 0.0033 |
| noise_sigma10/A | all | 0.5988 → 0.8355 | +0.2367 [0.1406, 0.3370] | 0.0028 → 0.0034 | +0.0007 [-0.0009, 0.0025] | 0.8550 | 0.0044 |

### §11: diagnostics by actual routing correct/incorrect
- Correct/incorrect is determined by comparing the final selection with the reference target stream; the sector-indistinguishable penalty is not counted as an actual routing error. The following are raw observation counts without natural weighting; the noise conditions contain 10 repetitions, so the same sample can contribute several observations. Predicted entities are the normalised value multiset; D_T/D_I are the numbers of distinguishable target/interferer gold values hit.
- The four binding classes follow the original protocol (all target values hit vs at least one interferer value hit); neither does not mean that no target value was hit, and may include partial target hits.

| Condition/abstention | Actual routing | Observations | Total predicted entities P2→LoRA | Mean per observation P2→LoRA | D_T P2→LoRA | D_I P2→LoRA | Four-class counts P2→LoRA (target only/interferer only/both/other) |
|---|---|---:|---:|---:|---:|---:|---|
| C5/none | correct | 199 | 196 → 237 | 0.9849 → 1.1910 | 124 → 184 | 0 → 0 | 104/0/0/95 → 159/0/0/40 |
| C5/none | wrong | 1 | 1 → 1 | 1.0000 → 1.0000 | 0 → 0 | 0 → 1 | 0/0/0/1 → 0/1/0/0 |
| sector_K8_center/none | correct | 163 | 160 → 193 | 0.9816 → 1.1840 | 101 → 150 | 0 → 0 | 84/0/0/79 → 130/0/0/33 |
| sector_K8_center/none | wrong | 37 | 36 → 44 | 0.9730 → 1.1892 | 0 → 0 | 26 → 35 | 0/26/0/11 → 0/31/0/6 |
| sector_K8_center/A | correct | 113 | 112 → 130 | 0.9912 → 1.1504 | 71 → 101 | 0 → 0 | 64/0/0/49 → 92/0/0/21 |
| sector_K8_center/A | wrong | 5 | 5 → 5 | 1.0000 → 1.0000 | 0 → 0 | 4 → 4 | 0/4/0/1 → 0/4/0/1 |
| sector_K8_center/B | correct | 125 | 124 → 145 | 0.9920 → 1.1600 | 81 → 114 | 0 → 0 | 72/0/0/53 → 103/0/0/22 |
| sector_K8_center/B | wrong | 0 | 0 → 0 | — | 0 → 0 | 0 → 0 | 0/0/0/0 → 0/0/0/0 |
| noise_sigma10/none | correct | 1731 | 1703 → 2044 | 0.9838 → 1.1808 | 1076 → 1586 | 0 → 0 | 916/0/0/815 → 1390/0/0/341 |
| noise_sigma10/none | wrong | 269 | 262 → 335 | 0.9740 → 1.2454 | 0 → 0 | 173 → 260 | 0/173/0/96 → 0/218/0/51 |
| noise_sigma10/A | correct | 1195 | 1167 → 1382 | 0.9766 → 1.1565 | 748 → 1067 | 0 → 0 | 660/0/0/535 → 963/0/0/232 |
| noise_sigma10/A | wrong | 19 | 18 → 21 | 0.9474 → 1.1053 | 0 → 0 | 12 → 17 | 0/12/0/7 → 0/15/0/4 |

### Summary of the frozen §6 gates

| Gate | P2 | LoRA | Conclusion |
|---|---:|---:|---|
| C1 selection-half target hit ≥ P2−0.03 | 0.636 | 0.860 | pass (existing) |
| C1 selection-half intent ≥ P2−0.03 | 0.687 | 0.897 | pass (existing) |
| C1 selection-half parse failures ≤ 2% | — | 0 | pass (existing) |
| C4-direct target hit ≥ P2−0.03 | 0.670 | 0.875 | pass (existing) |
| C3 target hit ≥ P2−0.03 | 0.125 | 0.180 | pass (existing) |
| C3 interferer-only ≤ P2+0.02 | 0.115 | 0.220 | **fail (original verdict kept)** |
| C5_pair_flip ≥ P2 | 0.30000 | 0.63000 | pass |
| sector_K8_center_none ≤ P2+0.01 | 0.05051 | 0.05882 | pass |
| sector_K8_center_A ≤ P2+0.01 | 0.00924 | 0.00924 | pass |
| sector_K8_center_B ≤ P2+0.01 | 0.00000 | 0.00000 | pass |
| noise_sigma10_none ≤ P2+0.01 | 0.03151 | 0.03995 | pass |
| noise_sigma10_A ≤ P2+0.01 | 0.00276 | 0.00343 | pass |

- Remaining final-SpatialNet gates: all pass. **Overall §6 still fails**; the original C3 conclusion is unchanged. Thresholds, adapters and the experimental protocol are not changed.
- C1-R is a pre-specified reporting condition with no additional gate. Whisper+T2-LoRA has not yet been started; the final SLURP test has not been run.

### Interpretation of results and job verification
- Final LoRA inference 22029676/22029677 both COMPLETED/0:0 (9:54, 8:56), 200 items each, ALL_DONE; scoring 22030055 COMPLETED/0:0 (23 s); paired analysis 22030132 COMPLETED/0:0 (4 s), log INPUTS_AND_FROZEN_METRICS_PASS. No new failures or retries.
- C5 target hit +0.215 [0.150,0.275], report half +0.200 [0.110,0.290]; pair flip 0.300→0.630. For the only actually misrouted sample g038_s1, the number of predicted entities is still 1, but interferer-value hits go 0→1; for the 199 correctly routed samples interferer hits stay at 0, and target-value hits go 124→184.
- The rise in interferer hits for C6 without abstention also occurs entirely in the actually misrouted group: among the 37 misrouted observations for 8 sectors, predicted entities 36→44 and D_I 26→35; among the 269 misrouted repeated observations for σ10, predicted entities 262→335 and D_I 173→260. D_I in the correctly routed groups is 0 under every main condition/abstention strategy, and target entity hits increase; both is 0 among the four classes.
- These observations are consistent with "the model is better at extracting entities from the selected speech, so the consequences of existing wrong selections are more visible"; the rise in interferer-only cannot be directly interpreted as a degradation of the LoRA's spatial binding ability. Routing itself is exactly the same for the two models; but the increased risk of the system answering with the interferer is a real observation and cannot be erased. C5 has only 1 independent misrouted sample, and noise repetitions are not new independent samples, so this cannot prove that there is generally no binding degradation.
- The gates look only at frozen point estimates: 8 sectors no abstention, naturally weighted interferer difference +0.00831, 95%CI [-0.00037,0.01894]; σ10 no abstention +0.00844, [0.00397,0.01369]. Both point estimates are ≤0.01, so they pass, but the upper bounds of the intervals exceed 0.01; **it is not statistically proven that the risk increase is below 0.01**.
- 8 sectors abstention B: interferer-only 0 for both models, coverage 0.86155; LoRA target hit 0.84448. σ10 abstention A: LoRA interferer-only 0.00343, coverage 0.85504, target hit 0.83545. Main stream-selection errors 0 and 0.00436 respectively.

## 26. Pre-start checks and configuration for step 2 (2026-11-07; written before training results)
- The project lead has approved in principle executing step 2 as established: it may be submitted after the prerequisite data, environment, budget and duplicate-job checks pass; on failure/timeout/budget overrun/protocol conflict, stop, with no self-initiated configuration change and retry.
- The training and selection rules of §15 are unchanged: Whisper transcripts of the same 7026 train audio items + frozen T2, target transcript = original SLURP text; 112 q/k/v/o modules of the thinker language layers, r16/alpha32/dropout0.05, bf16, gradient checkpointing, AdamW wd0, clip1, micro4×acc4, 3epoch×440=1320 steps, warmup40, lr={5e-5,1e-4}, seed2026, checkpoint220/440/660/880/1100/1320. Each epoch shuffles the same audio-key order as step 1 with Random(2026+epoch).
- Selection only on the Whisper transcripts of the 214 recordings/191 sentences of the C1 selection half by "at least one gold value hit", ties to fewer steps, then the smaller lr; C1-R, mixture conditions and the report half take no part in selection. The comparison direction is still Audio LoRA−Whisper+T2-LoRA, B2000/seed2026, single-talker by slurp_id, mixture by angle-stratified paired group bootstrap. The C3 failure and the C6 non-inferiority limits are kept.
- Transcript source `outputs/lora_base_audio/whisper_train_all/part*of6_2202424*/W/raw.jsonl`, all six jobs 22024240–45 COMPLETED/0:0; the 7026 keys agree with step 1, train_si284 follows the original list, and nothing is re-transcribed/no audio regenerated. The training entry point additionally checks the SLURP gold labels item by item and records input SHA256; only existing split metadata are used for the leakage audit, and no final test inference is done.
- Environment `<SHARED_ROOT>/envs/qwen_omni_lora`: torch2.6.0+cu124, transformers4.52.4, peft0.15.2, accelerate1.7.0, safetensors0.8.0; environment not changed. Shared disk available 127GB (88%); new checkpoints and resume states expected about 1.5–2GB, no new audio.
- The GPU cap is the 24h revised in §17. sacct accounting of 63 LoRA/supporting baseline GPU jobs (including old failures/timeouts): 12.86056h used, 11.13944h remaining; details in `outputs/lora_step2_preflight/20261107/gpu_budget_before.json`. The SpatialNet final front end and original-model reruns are not counted; the two final audio LoRA streams are counted. There is currently no LoRA running/queued and no step2 directory; no duplicate submission.
- New entry point `train_step2_lora.py/.sbatch`: in a separate file it reuses the step-1 training/resume implementation and the step-0 text encoding/answer loss, only replacing the audio input with the Whisper input, keeping the data order/the last small batch of 2 items/the six checkpoints. Overwriting segment provenance or an existing first-segment directory is prohibited; resumption additionally checks that the input SHA256 agrees.
- Submission configuration: <SLURM_ACCOUNT>/<GPU_PARTITION>, 1 A100-PCIE-40GB, 4CPU, 48GB, 30 minutes; soft stop and save at 24 minutes, guard at 27 minutes; resubmit after checking each segment, no automatic submission. First a CPU check of all 7026 input tokens/gold labels (<SLURM_ACCOUNT>/<CPU_PARTITION>, 4CPU/16GB/20 minutes), then a separate 40-step speed pilot (same basis as the step-1 pilot, not used for formal training), and after it passes, formal training of the two settings. Each segment starts with the original exact resume verification; no retry on failure.
- New paths `$SHARED/checkpoints/lora_v2/step2/lr{5e-5,1e-4}/`, pilot `step2_pilot/<job>/`; inference `outputs/lora_step2_eval/` (one checkpoint per job, 1GPU/4CPU/48GB/20 minutes). Estimated formal training of the two settings 2.6–3h, plus 12 selection-half checkpoint inferences and the final conditions about 1.4–2.5h, total increment 4–5.5GPUh; re-estimate after the pilot; stop whenever the projection exceeds 24h, without reducing candidates.
- The GPUs already allocated to other jobs are neither cancelled nor taken; other training and scheduling priorities are not changed; independent resources allocated normally by SLURM are used, and queueing time cannot be guaranteed. Other training such as TFGridNet is left as it is.

### Step 2 prerequisite CPU checks: one passed, one failed to start (not retried)
- Full training-input check **22030203**: COMPLETED/0:0, 2:06; DATA_AUDIT_PASS 7026, ALL_INPUT_IDS_MATCH_INFERENCE 7026, TOKEN_LENGTH_MAX 671, DRY_RUN_PASS. Run at commit90553eb, no dirty dependencies. No GPU model loaded, no training.
- Six-condition inference input check **22030260**: FAILED/2:0, 1 s; the log is `python: can't open file '/tmp/step2_inference_dryrun.py': [Errno 2] No such file or directory`. The submission driver was in the login node's local /tmp, which is not visible on compute nodes, so the check script never started. This is a submission-path error, not a data/model/scoring failure; the GPU budget did not increase.
- Following the project lead's requirement for this round, "stop on failure; do not change the configuration and retry on one's own initiative", the dependent step-2 steps were stopped; **neither the pilot nor formal GPU training was submitted, and the CPU job was not rerun**.
- A reviewable fix was prepared: the same CPU driver was added to the shared repository as `scripts/lora/step2_inference_dryrun.py`, locating the project with `Path(__file__).resolve().parents[2]`. Only the visibility of the script path is fixed; inference logic, experimental configuration and frozen rules are unchanged; the syntax check passes; not yet rerun on a compute node.
- The rerun awaiting approval keeps <SLURM_ACCOUNT>/<CPU_PARTITION>, 4CPU, 16GB, 20 minutes, running that repository script under a new job ID/separate log; after it passes, the approved speed pilot and formal step 2 are executed. The failure log `outputs/logs/step2_eval_dryrun_22030260.out` is kept. All existing experimental results and the final test remain unchanged.

### Approved CPU pre-check rerun passed (2026-11-07)
- Using commit533f899, job **22030284** (4CPU/16GB/20 minutes) COMPLETED/0:0, 1:22; compute node <NODE> confirmed that the driver, training ASR data, lists, model configuration and environment are all accessible. Separate output `outputs/lora_step2_preflight/retry_533f899/` (access.json, full log, result.json).
- Both halves of C1/C1-R with 214 each and both final streams with 200 each: six items DRY_RUN_PASS, finally ALL_SIX_INFERENCE_DRYRUNS_PASS; output complete. The old failure 22030260 is not overwritten.
- Prerequisite checks pass; per the approved plan, continue with the separate 40-step GPU pilot (5e-5, original 1320-step schedule), update the budget after checking, and start formal training. The pilot directory is not used for checkpoint selection.

### Step 2 pilot submitted, waiting for GPU resources (checked 2026-11-07 23:47 EST)
- **22030353**: `sbatch scripts/lora/train_step2_lora.sbatch 5e-5 1 pilot 40`; 1 A100/4CPU/48GB/30 minutes. Currently PENDING, Reason=Priority, RunTime=0; no GPU training results, no formal step-2 training or checkpoint-selection results.
- squeue --start provisionally estimates a start at 2026-11-08 01:48:05 EST (a scheduler prediction that may change; not used to judge completion). No configuration/priority changes, no duplicate submission, no intervention in other jobs.
- Expected pilot output `$SHARED/checkpoints/lora_v2/step2_pilot/22030353/`; log `outputs/logs/lora_step2_train_22030353.out`. After the pilot completes, sacct/log, 40 contiguous finite losses, resume saving and reference steps, speed and GPU memory must be checked and the 24h budget re-estimated; only if these pass is segment 1 of the two formal settings submitted.
- Cumulative GPU is still 12.86056h (the pilot has not run yet); formal training has not been submitted. CPU pre-check outputs and documents have been committed, old failure records kept; no new failures, no use of the final test.

### Step 2 pilot verified; entering formal training (2026-11-08)
- 22030353 COMPLETED/0:0, 2:46; the 40-step log is contiguous, loss/entity loss/gradients all finite; the adapter's 224 tensors and file length complete, the resume.pt archive CRC passes, reference steps 41/42 present, provenance not dirty. The pilot's SEGMENT_DONE or finished=False was not taken as formal training completion.
- Median 3.208 s per step, loading 25.6 s, training peak allocated18.86GiB/reserved21.03GiB; the two settings' 2640 steps are projected at 2.35GPUh of pure training, about 2.6–3h with segment overhead; all of step 2 is still projected at 4–5.5GPUh. Current cumulative 12.90667GPUh; the 24h budget is sufficient.
- The formal step2 directory does not exist, and there are no duplicate LoRA jobs queued/running; segment 1 of lr5e-5 and lr1e-4 submitted with the previously approved configuration, each 1A100/4CPU/48GB/30 minutes. Every subsequent segment must have sacct, log, state.json, checkpoints and the exact resume check verified before resubmission; training is judged complete only when both settings have 1320 steps and all six checkpoints.
- Formal segment-1 job IDs: **22033027** (5e-5), **22033028** (1e-4); submission code 9d32b2d, working tree clean before submission. Segment or training completion not yet judged.

### Verification of formal step-2 segment 1 (2026-11-08)
- 22033027 (5e-5): COMPLETED/0:0, 24:42, contiguous steps 1–406; 22033028 (1e-4): COMPLETED/0:0, 24:27, contiguous steps 1–407. Both state.json files are segment1/finished=False; overall training not yet complete.
- In both settings loss/entity loss/gradients are all finite; ckpt_step220 and the resume adapter have 224 tensors and complete file lengths, the resume.pt archive CRC passes, the following two reference steps are present, provenance not dirty. The next segment must still perform the established exact resume check.
- Median per step 3.269/3.305 s, training peak allocated18.88GiB/reserved21.03GiB; cumulative GPU 13.72583h, 10.27417h remaining; completing the remaining training and evaluation is still projected within the 24h budget. No duplicate LoRA jobs running/queued; continue with segment 2 under the original configuration.
- Segment 2 submitted: 22049407 (5e-5, from step 407), 22049408 (1e-4, from step 408), submission code 4e92867, working tree clean; waiting for the jobs and resume verification, without pre-judging a pass.

### Verification of step-2 segment 2 (2026-11-08)
- 22049407 (5e-5) COMPLETED/0:0, 24:43, to step 815; 22049408 (1e-4) COMPLETED/0:0, 24:17, to step 818. Both states segment2/finished=False; overall training not complete.
- Both exact resume checks of global_step/LoRA checksum/optimizer state/next_lr/data_order pass. Logs 1–815 and 1–818 contiguous without duplicates; loss/entity loss/gradients all finite; the step220/440/660 adapters and the resume adapter have 224 tensors and complete lengths; the resume.pt archive CRC and the records of the next two reference steps pass.
- Cumulative GPU 14.54250h, 9.45750h remaining; measured 3.255/3.249 s per step, training GPU memory allocated18.88GiB, no new failures. Segment 3 submitted with the original configuration, without changing hyperparameters or frozen rules, not involving Project 1 or the final test.
- Segment-3 jobs 22049681 (5e-5, from step 816) and 22049682 (1e-4, from step 819) submitted; submission code e6e6997, working tree clean. Overall completion and the best checkpoint still await later verification and the frozen selection evaluation.

### Verification of step-2 segment 3 (2026-11-08)
- 22049681 (5e-5) COMPLETED/0:0, 24:43, to step 1224; 22049682 (1e-4) COMPLETED/0:0, 24:44, to step 1227. Both states segment3/finished=False; overall training not yet complete.
- Both exact resume checks pass; logs contiguous, losses and gradients finite; the step220/440/660/880/1100 adapters and the resume adapter have 224 tensors/complete file lengths; the resume.pt archive CRC and the records of the next two reference steps pass.
- Cumulative GPU 15.36667h, 8.63333h remaining; about 3.236/3.235 s per step. Segment 4 submitted with the original configuration to finish, without changing the training or selection rules.
- Segment-4 jobs 22050126 (5e-5, from step 1225) and 22050127 (1e-4, from step 1228) submitted, submission code eea8c3d, working tree clean. 1320 steps/finished=True and all six checkpoints must be confirmed before starting the frozen selection evaluation.

### Step 2 selection-half checkpoint evaluation and frozen selection (2026-11-09)
- Under the frozen rules of §15, the 12 checkpoints of the two learning-rate settings were evaluated on the same Whisper + T2 C1 selection-half inputs (214 items, 191 `slurp_id`); the reporting half, C1-R, C5, C6 and the final test were not read.
- Pre-submission checks pass: the two Whisper inputs have 214 items each, 428 items and 382 `slurp_id` in total; all 12 training checkpoint paths exist; no duplicate evaluation jobs; each job uses `<SLURM_ACCOUNT>/<GPU_PARTITION>`, 1×A100, 4 CPU, 48 GB, 20 minutes; about 126 GB remaining on the shared disk; worst case about 4 GPU hours, the budget still below 24 hours.
- Evaluation jobs 22052665–22052676, all `COMPLETED/0:0`, taking 7:17–12:25, separate outputs in `outputs/lora_step2_eval/`. Each output has exactly 214 items and 191 `slurp_id`, the logs contain `ALL_DONE`, and the input fields agree with the frozen Whisper baseline; no OOM, timeout or configuration mismatch.

| lr | step 220 | 440 | 660 | 880 | 1100 | 1320 |
|---|---:|---:|---:|---:|---:|---:|
| 5e-5 | 0.818 | 0.818 | 0.822 | 0.808 | 0.813 | 0.818 |
| 1e-4 | 0.813 | 0.822 | **0.832** | 0.827 | 0.808 | 0.799 |

- The primary metric is "at least one gold value hit". The unique best checkpoint is **lr=1e-4, step=660**, selection-half metric **0.8318 (178/214)**; the non-fine-tuned Whisper + T2 baseline is **0.6075 (130/214)**, difference **+0.2243**. The other, tied or lower candidates are not taken under the frozen rules; no step or learning-rate tie-breaking was triggered this time.
- Selection-half paired bootstrap (grouped by `slurp_id`, B=2000, seed=2026): difference 0.2243, 95% interval **[0.1521, 0.2996]**, lower bound > 0; parse failures 0/214. The training-loss check passes (mean of the first 37 steps 0.18899, mean of the last 110 steps 0.02812); no "loss did not decrease" exception. The gate result is written to `outputs/lora_step2_eval/selection_gate_22052665_76/gate.json`, conclusion `pass`.
- Other selection-half descriptive metrics of the best checkpoint: date recall 0.8571 (54/63), strict intent accuracy 0.8832, parse failures 0, gold entities hit 229/298. These are descriptive only and take no part in checkpoint selection.

### Step 2 post-freeze evaluation (2026-11-09)
- Uses the unique frozen checkpoint `step2/lr1e-4/ckpt_step660`; no checkpoint was reselected and the test split was not used. GPU jobs 22053122 (C1 report), 22053123 (C1-R all), 22053124 (SpatialNet ep100 C5 selected route), 22053125 (C5 other route) all `COMPLETED/0:0`, taking 6:13–14:22, all 1×A100/4 CPU/48 GB/20 minutes; the outputs have 214, 428, 200 and 200 items respectively, and the logs contain `ALL_DONE`.
- Input consistency passes: C1/C1-R use the existing Whisper transcripts; both C5 streams use the same batch of processed SpatialNet ep100 audio and the existing W outputs, with `file/group/swap/target/ref_*` and `asr_text` checked item by item; the prompts, target definitions, `slu-score-v2`/`binding-score-v1` scoring and checkpoint provenance of the two routes are consistent. Per-sample outputs in `outputs/lora_step2_eval/lr1e-4/step660/`, summaries in `outputs/lora_step2_eval/c1_report_22053122/`, `c1r_22053123/`, `final_checks_22053124_25_v3/`.

#### C1 and C1-R (single-talker)
- Primary metric "at least one gold value hit"; paired bootstrap grouped by `slurp_id`, B=2000, seed=2026:

| Condition | LoRA | Whisper+T2 | Difference [95% CI] |
|---|---:|---:|---:|
| C1 reporting-half | 0.8364 | 0.5748 | +0.2617 [0.1864, 0.3391] |
| C1-R reporting-half | 0.7430 | 0.4813 | +0.2617 [0.1905, 0.3364] |
| C1 all 428 | 0.8341 | 0.5911 | +0.2430 [0.1885, 0.2960] |
| C1-R all 428 | 0.7383 | 0.5023 | +0.2360 [0.1868, 0.2847] |

- C1 report-half LoRA parse failures 0/214 (baseline 2/214); C1-R all parse failures 0/428 (baseline 3/428). C1-R is only a pre-specified report and is not used for checkpoint or threshold adjustment.

#### SpatialNet ep100 C5 and the frozen gates
- C5 selected stream, 200 items: target-any **0.745 vs 0.570** (LoRA vs Whisper+T2), naturally weighted paired difference **+0.1271 [0.0642, 0.1960]**; pair-flip **0.48 vs 0.30**. Naturally weighted interferer-only **0.00354 vs 0.00159**, difference +0.00196 [0, 0.00587].
- The C5 gate `pair_flip >= baseline` passes. The point estimates of the main C6-real gates also pass: 8 sectors no abstention interferer-only 0.05723 vs 0.04733 (difference +0.00990 ≤ 0.01), σ10 no abstention 0.03609 vs 0.02827 (difference +0.00782 ≤ 0.01); the corresponding gates for abstention A/B all pass. The existing C3 interferer-only gate failure is kept, and the overall §6 conclusion does not change.

#### C6δ and c6-real-v1
- C6δ uses the frozen ±10°/±20° query offsets; the fixed abstention threshold keeps the established ep100 `τ=16°`. Without abstention, the naturally weighted target-any differences LoRA−baseline [95% CI] are: +10° **+0.1141 [0.0524, 0.1837]**, −10° **+0.1188 [0.0569, 0.1877]**, +20° **+0.0870 [0.0283, 0.1543]**, −20° **+0.1149 [0.0524, 0.1836]**. The interferer-only differences are +0.00794, +0.00831, +0.02523 and +0.01418 respectively.
- With τ=16°, C6δ eval-half coverage (LoRA / baseline): +10° 0.54/0.54, −10° 0.74/0.74, +20° 0.55/0.55, −20° 0.68/0.68; the corresponding LoRA target-any is 0.759, 0.784, 0.600, 0.794; the abstention threshold was not changed.
- c6-real-v1: naturally weighted coverage for 8 sectors abstention A/B is LoRA 0.846/0.862 and baseline 0.846/0.862; for σ10 abstention A, LoRA 0.855 and baseline 0.855. The naturally weighted main stream-selection error rate and interferer-only risk are reported by the frozen scripts. All intervals use the pre-specified group-level/paired bootstrap (B=2000, seed=2026); the frozen protocol has no additional multiple-comparison correction item, so no correction was added and no gate adjusted.
- Provenance of the statistics and per-sample results is in `outputs/lora_step2_eval/final_checks_22053124_25_v3/checks.json`, `c6delta_baseline_t2_v2/`, `c6delta_lora_t2_v2/`; the analysis scripts only extended the existing frozen evaluation scripts to read the T2-LoRA route, without changing prompts, data, scoring or thresholds.
## Audio LoRA vs Whisper + T2-LoRA comparison (development-set, 2026-11-09)

This section uses only the frozen Audio LoRA step 440 and Whisper+T2-LoRA `lr=1e-4, step=660`; the final test is not used and no checkpoint is reselected. All statistics reuse the frozen scorer, sample order, thresholds and paired bootstrap (B=2000, seed=2026). Per-sample inputs and outputs are saved in `outputs/lora_step2_eval/audio_vs_t2_comparison_v2/`, the C5/C6 statistics in `outputs/lora_step2_eval/audio_vs_t2_final_checks/checks.json`.

### Revision of the statistical direction (2026-11-09)

A check found that the paired fields of the original `audio_vs_t2_final_checks/checks.json` followed the historical `candidate−baseline` order, i.e. what was actually computed is Whisper+T2-LoRA − Audio LoRA; the original per-sample predictions, scoring and CI values themselves are fine. The table header, however, said Audio−T2, so the sign was reversed. This is a statistical-direction/report-label error, not a data error, and it changed neither the scoring protocol, thresholds nor historical gates.

The explicit Audio−T2 results were recomputed from the same per-sample outputs; the revised file is `outputs/lora_step2_eval/audio_vs_t2_final_checks_corrected/checks.json`. The revised C5 raw difference is +0.0900 [0.0500,0.1300], natural-weighted +0.1167 [0.0521,0.1914]. The raw differences of C6δ +10/−10/+20/−20 are +0.0700/+0.0800/+0.0600/+0.0750; the corresponding natural-weighted differences are +0.1097/+0.1136/+0.1057/+0.1116, with all CIs consistent with the per-sample recomputation. The old output is kept as pre-revision provenance and not overwritten.

## Cross-Dataset Generalization design (plan only, not started)

The goal is to test generalisation of spoken commands beyond SLURP without training the existing three routes. The main validation set is changed to the public **Fluent Speech Commands (FSC)**: it provides independent speakers, recordings and spoken-command expressions, suitable for testing cross-dataset transfer. **MASSIVE is no longer the "fully independent" main validation set**; since its task scheme and semantic content may inherit from/overlap with SLURP, it can only serve as a supplementary analysis with strict text, intent and speaker overlap audits, and cannot on its own support a conclusion of independent cross-domain generalisation.

To avoid label-scheme inconsistencies, before evaluation each sample is mapped to a frozen common JSON: `transcript`, `intent_family`, `slots` (unified as normalised type/value), and only the mappable common semantic subset is reported. Primary metrics are target-any, target-all, intent accuracy, slot precision/recall, parse failure, abstention coverage and answer risk; all systems use the same prompt output format and the same scorer, and the original labels and mapping table are kept for auditing.

Audio conditions are in four layers: single-talker clean original audio; single-talker reverberation with real or simulated room impulse responses; two-talker spatialised mixtures (two speakers, random azimuths and SIR strata); direction uncertainty (target DOA ±10°/±20°, sector K8 and σ=10° perturbation). The two-talker conditions keep the existing SpatialNet manifests, sample rate and route definitions, replacing only the corpus audio; SpatialNet is not retrained.

The three frozen routes compared are original Qwen audio input, Audio LoRA P2, and Whisper+T2-LoRA T2. Each condition uses the same audio, the same sample order, the same number of repetitions and paired bootstrap; results stratified by dataset are reported first, then pooled cross-dataset results. If the advantage on SLURP disappears while performance drops on the independent dataset, it is flagged as possible SLURP-specific overfitting, and checkpoints or thresholds are not adjusted on that basis.

Data availability: FSC as the main validation set, MASSIVE only as a supplement; for both, the version, download date, SHA256 and speaker list must be recorded. Everything is transcoded to 16 kHz mono PCM; spatialisation uses the frozen RIR/SpatialNet inference chain, saving the mixture audio, DOA, SIR and route manifests. CPU-side parsing and statistics need no GPU; the inference budget is estimated as three models × datasets × four conditions, preferably reusing existing inference scripts and weights in batches, requesting A100 jobs only after confirming that outputs are missing.

The main leakage risks include: pretrained speech models may have seen public corpora; MASSIVE/Fluent may share templates or entities with SLURP; the same speaker may appear in several splits; and SpatialNet/RIR generation parameters may be exposed by the training set. The plan requires speaker deduplication, near-duplicate text retrieval, entity vocabulary audits, and listing all cross-dataset overlapping samples in an exclusion list; the final test is still not used at this stage.

| development split / condition | Audio LoRA | Whisper+T2-LoRA | Audio−T2 paired delta (95% CI) |
|---|---:|---:|---:|
| C1 reporting-half target-any (n=214) | 0.8925 | 0.8364 | +0.0561 [+0.0183,+0.0995] |
| C1-R reporting-half target-any (n=214) | 0.8364 | 0.7430 | +0.0935 [+0.0439,+0.1469] |
| C5 raw target-any (n=200) | 0.8350 | 0.7450 | +0.0900 [+0.0500,+0.1300] |
| C5 natural-angle target-any | 0.8384 | 0.7217 | +0.1167 [+0.0521,+0.1914] |
| C6δ +10° raw target-any | 0.6750 | 0.6050 | +0.0700 [+0.0350,+0.1050] |
| C6δ −10° raw target-any | 0.7100 | 0.6300 | +0.0800 [+0.0400,+0.1200] |
| C6δ +20° raw target-any | 0.5100 | 0.4500 | +0.0600 [+0.0250,+0.0950] |
| C6δ −20° raw target-any | 0.6500 | 0.5750 | +0.0750 [+0.0350,+0.1150] |

C5 pair-flip is Audio 0.630, T2 0.480; wrong-speaker-only is 0.005/0.010 (equal samples), naturally weighted 0.00159/0.00354. Neither has parse failures or abstentions on C5; Audio has `M=177,U=16`, T2 `M=159,U=28`. C6-real-v1 naturally weighted only-interferer: sector K8 center 0.05882/0.05723, Audio−T2 paired difference −0.00159 [−0.00941,+0.00587]; noise σ=10° 0.03995/0.03609, difference −0.00386 [−0.00924,+0.00174]. Abstention/coverage and interferer hits are therefore reported separately, and conclusions cannot rest on target-any alone; the C6 gates keep the frozen criteria. Overall §6 still fails because of the pre-existing C3 gate failure; this comparison does not change that conclusion.

After a per-sample check of the raw C5 outputs of the two routes, 93 items show output differences; the evidence labels are written to `disagreements.jsonl`. The labels indicate evidence only, not causal conclusions: samples where the Whisper transcript differs from the reference sentence are labelled "transcription/input difference evidence", and the other output differences are labelled audio LLM entity/intent understanding differences; T2 outputs do not carry the physical stream field of P2, so the missing field was not misreported as a stream-selection error. The current file contains no evidence that could on its own prove separation distortion or annotation-span errors; classification would require manual listening/annotation audits.

Input verification: C1/C1-R use the same 214 `slurp_id`; C5 uses the same 200 `g###_s{1,2}`, the same SpatialNet ep100 selection results, target/interferer definitions and scorer. The audio SHA of Audio P2 agrees with its frozen job configuration; the `asr_text` of T2 comes from the same frozen Whisper input; prompts, checkpoint provenance and output counts are all recorded. All of the above are development-set results and cannot be taken as final test results.

### Completing the primary comparison: C1 / C1-R selection half and all (2026-11-11, CPU, decision of the project lead)
- Completed per §15 item 4. The rules are exactly the same as for the report half: binding-score-v1 "at least one gold value hit" (empty interferer set), paired bootstrap grouped by slurp_id, B = 2000, `np.random.default_rng(2026)`; `hit()` / `boot()` / `rows()` imported directly from `scripts/lora/compare_audio_vs_t2.py`, unchanged. Script `scripts/lora/compare_audio_vs_t2_halves.py` (login-node CPU, no job).
- Inputs (existing per-sample outputs, no new inference): Audio LoRA `outputs/lora_audio_eval/lr1e-4/{22025280/step440/devel_clean_selection, 22027861/…/devel_clean_report, 22027862/…/devel_rev_selection, 22027863/…/devel_rev_report}/raw.jsonl`; Whisper + T2-LoRA `outputs/lora_step2_eval/lr1e-4/step660/{devel_clean_selection_22052673, devel_clean_report_22053122, devel_rev_all_22053123}/T2/raw.jsonl`. Assertions: 214 items per half, 428 items / 382 slurp_ids in total, no intersection between the halves, slurp_ids and gold entities identical item by item for the two routes; the report-half results agree field by field with the original `comparison.json`.
- Results written to `outputs/lora_step2_eval/audio_vs_t2_comparison_v2/comparison.json` (adding `C1_selection`, `C1_all`, `C1R_selection`, `C1R_all`, with existing keys unchanged; the file before modification was copied to `comparison_before_halves.json` in the same directory).

| Condition | Half | Audio LoRA | Whisper+T2-LoRA | Audio−T2 [95% CI] |
|---|---|---:|---:|---:|
| C1 | selection | 0.8598 | 0.8318 | +0.0280 [−0.0095, +0.0670] |
| C1 | report | 0.8925 | 0.8364 | +0.0561 [+0.0183, +0.0995] |
| C1 | all | 0.8762 | 0.8341 | +0.0421 [+0.0139, +0.0709] |
| C1-R | selection | 0.7897 | 0.7336 | +0.0561 [+0.0046, +0.1053] |
| C1-R | report | 0.8364 | 0.7430 | +0.0935 [+0.0439, +0.1469] |
| C1-R | all | 0.8131 | 0.7383 | +0.0748 [+0.0376, +0.1132] |

- External judgement still looks only at the report half (unchanged). On the selection half, the checkpoints of both routes were selected on the C1 selection half, so the intervals are optimistic; the C1 selection-half interval crosses 0, which is recorded as is and changes no verdict.

### Note on the §6 verdict (2026-11-11, decision of the project lead; rules and thresholds not changed)
- The overall §6 verdict remains **fail (C3)**.
- Interpretation: on C3 (single-channel unseparated mixture + direction instruction), interferer-only 0.115 → 0.220 shows that the audio LoRA **has not learned to pick the talker by direction**; on unseparated mixture recordings it extracts entities more aggressively, including the interferer's entities.
- On the actual system path (C5 and c6-real-v1 after SpatialNet ep100 separation + v3 stream selection) all gates pass (§25); but for c6-real-v1 8 sectors without abstention (+0.0083 [−0.0004, +0.0189]) and σ10 without abstention (+0.0084 [+0.0040, +0.0137]), the upper bounds of the 95% intervals of the interferer-only differences both exceed 0.01; the gates look only at point estimates under the frozen rules, so it cannot be regarded as statistically proven that the risk increase is below 0.01.

### Note on old labels (2026-11-11)
- In `C5_disagreements.categories` of `outputs/lora_step2_eval/audio_vs_t2_comparison_v2/comparison.json`, **"选路差异": 93 is a labelling error of an old script version**: the audio LoRA and Whisper+T2-LoRA use the same selected stream (the same audio) on C5, so there is no stream-selection difference between them; the count merely applies that label to all 93 disagreeing samples. Likewise "Whisper转写/输入差异证据": 93 is a blanket label on all disagreeing samples and cannot be read as all 93 having substantive transcription errors (most may differ only in punctuation or case).
- These three category counts are descriptive only and enter no metric, interval or verdict; the file was not modified; this note takes precedence.
