English translation of slurp_test_freeze_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：slurp_test_freeze_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# SLURP test freeze document v1 (approved 2026-11-11; not yet run)

> - Basis: `docs/lora_slu_plan_v2.md` §9 (test is run only once, on the final frozen configuration; a separate document freezes it before the run), `docs/lora_v2_runs.md`, `docs/spatialnet_final_v1.md`, `docs/c6_real_v1.md`, `docs/mix_spec_v1.md`.
> - When this draft was written, **no test audio, test annotation content or test_eval92 RIR had been read**; the test sizes below only cite the split statistics already recorded in `docs/log/day1_followup4.md` §9.
> - Finalised from the draft `docs/slurp_test_freeze_v1_draft.md` (kept unchanged). All open items of the draft were **decided by the project lead on 2026-11-11**, see §0; the commit that adds this file is the freeze point.
> - After the freeze, **test audio is still not read** until the project lead separately approves the test execution steps. After approval, the CPU construction (§4) is done first and its sha256 values recorded, and only then is GPU inference submitted. After the test results come out, no tuning, no re-selection of checkpoints and no threshold changes; reruns follow only the failure-rerun rule of §0-8.

## 0. Approval decisions (project lead, 2026-11-11)
1. **The audio LoRA is included in test.** Reason: §6 fails only on C3; C3 is not the actual system path, and the actual path (C5, c6-real) passes; without including it the pre-specified primary comparison cannot be tested. On test, the §6 results and C3 are reported as they are, as a limitation.
2. **Primary comparisons**: three, C1, C1-R and C5 (audio LoRA − Whisper + T2-LoRA). **Bonferroni correction**: each reports a 98.33% paired bootstrap interval (0.8333% / 99.1667% quantiles), judged significant only if the lower bound is > 0; the uncorrected 95% interval is also reported. All other conditions are descriptive only.
3. **test mixture set**: constructed with the mix_v1 rules, **200 groups (400 samples)**, equal across the four angle strata (50 groups each); if fewer than 200 feasible groups are available, take the largest feasible number (keeping the four strata equal), no fewer than 100 groups; if fewer than 100, stop and report.
4. **C0** adds the T2-LoRA result on reference transcripts, descriptive only.
5. **Natural angle weighting**: the main results use the cv_dev93 distribution; a version with the distribution of test_eval92 itself is also reported, descriptive only.
6. **The original Qwen route** uses `envs/qwen_omni`; no comparison check between the two environments.
7. **The test GPU budget is separate, capped at 10 hours**; if the estimate exceeds 10 hours, switch to 100 groups and state so.
8. **Failure-rerun rule**: if a job fails because of a program error, timeout or node failure without producing complete output, it may be rerun with exactly the same configuration, with the reason recorded; a condition that has already produced complete output must not be rerun because of its results.
9. **Basis for judging C5 (confirmed by the project lead 2026-11-12)**: the C5 primary comparison is judged on the **equal-weight (raw) basis** (98.33% interval lower bound > 0); natural angle weighting is descriptive only, consistent with the devel primary comparison.

## 1. Frozen configuration

| Component | Frozen value |
|---|---|
| Base model | Qwen2.5-Omni-7B, commit `ae9e1690` (local snapshot, offline), bf16, sdpa, talker disabled, greedy decoding, `thinker_max_new_tokens=256` |
| Prompts | P2, T2 (`PROMPT_P2` / `PROMPT_T2` in `scripts/qwen/qwen_omni_c1_p2t2.py`, text unchanged); C3 direction instruction (`docs/mix_spec_v1.md`, frozen 2026-11-05) |
| ASR | Whisper large-v3, revision `06f233fe…` (`WHISPER_REV`), `language='en'`, greedy, `max_new_tokens=128` |
| Original Qwen | no adapter: original Qwen + P2 (audio route); Whisper + original Qwen + T2 (text route) |
| Audio LoRA | `checkpoints/lora_v2/step1/lr1e-4/ckpt_step440` (P2; r16/alpha32, 112 q/k/v/o modules of the thinker language model) |
| Whisper + T2-LoRA | `checkpoints/lora_v2/step2/lr1e-4/ckpt_step660` (T2) |
| Front end | SpatialNet-small ep100 final checkpoint: `<SHARED_ROOT>/checkpoints/final/spatialnet_ep100_20261107-223434/best.pt`, SHA256 `21caf0fc3b08cfc3b9860efbfeff9a756acecbe36101620378e47891edd76c10` |
| Stream selection | v3 (`eval_c6_delta.choose`: the stream whose estimated direction is closest to the query direction, stream 0 on a tie; zero-lag mic0 polarity correction) |
| Abstention | abstention A: abstain when margin < **τ = 17°** (fixed, c6-real-v1); abstention B: abstain when the estimated directions of both streams fall inside the query sector (C6-sector only) |
| Scoring | slu-score-v2, binding-score-v1 (normalize_v2), c1-error-taxonomy-v1, all unchanged |
| Environment | `<SHARED_ROOT>/envs/qwen_omni_lora` (torch 2.6.0+cu124, transformers 4.52.4, peft 0.15.2) for the LoRA routes; the original Qwen routes use `envs/qwen_omni` (same as devel, differing from the former only in peft; no comparison check between the two environments, §0-6) |

Each job records at run time: git commit (the working tree must be clean), adapter file SHA256, model snapshot revision, input audio SHA256, processed waveform SHA256.

## 2. Gate facts from the devel stage (written as they are, unchanged)
- The plan §6 degradation check **fails overall**, because of **C3**: interferer-only P2 0.115 → audio LoRA 0.220 (threshold ≤ P2 + 0.02). All other items (C1, C4-direct, C3 target hits, C5 pair flip, the five c6-real-v1 items) pass on point estimates; for c6-real-v1 8 sectors and σ10 without abstention, the upper bound of the interferer-only difference interval exceeds 0.01 (`lora_v2_runs.md` §25 and the note of 2026-11-11 §6).
- Interpretation (project lead 2026-11-11): the audio LoRA has not learned to pick the talker by direction; on unseparated mixture recordings it extracts interferer entities more aggressively; the actual system path (C5, c6-real) passes.
- **Whether to still include the audio LoRA in test: to be decided by the project lead.**
  - Decision: **include** (reason in §0-1); on test, the §6 results and C3 are reported as they are, as a limitation.
  - Decided by / date: project lead / 2026-11-11
- Primary comparison on devel (report half, Audio LoRA − Whisper + T2-LoRA): C1 +0.056 [0.018, 0.100], C1-R +0.093 [0.044, 0.147], C5 raw +0.090 [0.050, 0.130]. These are devel results and are not a prior threshold for test.

## 3. test conditions (each condition and each route is run only once)

| Condition | Input | Routes | Samples |
|---|---|---|---|
| **C1** | split_v2 test recordings, SLURP original 16 kHz, no level adjustment | original Qwen + P2; audio LoRA + P2; Whisper + T2; Whisper + T2-LoRA | all test recordings (recorded: 1,183 recordings / 812 sentences / 37 speakers) |
| **C0** | SLURP reference transcripts (original text) | original Qwen + T2; T2-LoRA + T2 (descriptive only, §0-4) | 812 sentences / 1,183 recordings (reported both per sentence and per recording) |
| **C1-R** | same recordings as C1, single-talker full reverberation: the `reverb_single.reverb()` chain with RIRs switched to **test_eval92** (assignment rule as on devel: sort by file name, `random.Random(2026).sample`, then `randrange(2)` per item to choose h_0/h_1) | same four as C1 | same as C1 |
| **C3** | test mixture, mic 0 single channel + frozen direction instruction + P2 | original Qwen + P2; audio LoRA + P2 | all samples of the test mixture set |
| **C5** | test mixture 6 channels → SpatialNet ep100 → v3 stream selection (query direction θ1) → `chain8(sign · out{k})` (8 kHz → `resample_poly(x,2,1)` → `level()` −30 dBFS, polarity applied only once) | same four as C1; inference on both the selected and the other stream | all samples of the test mixture set |
| **c6-real-v1 main conditions** | reuse the two C5 output streams; only the query direction and stream selection change, on CPU: 8-sector centre partition (K=8), σ = 10° noise (10 repetitions, `SeedSequence([2026, σ, r])`) | same as C5 | same as C5 |

- Not run: C4-direct, C6δ, K=4 / rotated partitions, σ=5°/15° (these are sensitivity analyses on devel only). The approval decisions did not include a sensitivity appendix, so they are not reported on test; reporting them would need separate approval and would be descriptive only, without judgement.

## 4. Construction of the test mixture set (generated on CPU before the run, with the same rules as mix_v1)
- Speech: split_v2 test recordings (basis A). RIRs: SMS-WSJ **test_eval92** (8 kHz, 6 channels, h_0/h_1).
- Pair filtering, sampling and mixing all follow `docs/mix_spec_v1.md`; only the split and the RIR set are replaced:
  1. same gender, duration difference ≤ 1 s, each recording used at most 3 times, D_T and D_I both non-empty, shared word stems / containment excluded (the rules of `devel_capacity_mixspec_v1.py`);
  2. at most 30 groups per speaker, at least 20 male groups;
  3. number of groups **200 (400 samples)** (§0-3); four angle strata [0,15), [15,30), [30,60), [60,180], 50 groups each, no repeated RIR sample; if infeasible, decrease in multiples of 4 to the largest feasible number (equal across the four strata), no fewer than 100 groups; below 100, stop and report;
  4. seed 2026: sort by (file_a, file_b), shuffle with `random.Random(2026)` to get random ranks, 0/1 integer programming maximising Σ(N_pairs − r)·x; if infeasible, do not relax the constraints; report the closest feasible result and stop;
  5. the same rng determines, in order, A/B directions, angle-stratum assignment, RIR sample within the stratum, offset, noise SNR ~ U(20,30) and noise seed;
  6. two swapped samples per group, both with query direction θ1; offset, rir_start, early RIR (50 ms), `extract_piece` alignment, mic 0 early-image active-frame RMS −30 dBFS (SIR 0 dB) and 6-channel white noise are all the same as in `mix_slurp_v1.py`.
- If a stratum has too few test_eval92 samples, stop and report; do not change the stratification.
- Implementation: the existing `make_mix_list_v1.py`, `mix_slurp_v1.py` and `reverb_single.py` currently accept only devel/train and cv_dev93/train_si284; test / test_eval92 options need to be added (adding branches only, without changing the behaviour of existing branches; rerun on devel to check that the existing lists and audio are byte-identical).
- After generation and before any inference, record and commit: the list JSON, the files of each sample directory, the C1-R audio, the SHA256 of the manifests; the angle distribution of the test mixture set and the natural angle distribution of test_eval92.
- **Natural angle weighting**: the main results use the cv_dev93 weights already frozen in c6-real-v1 (78/96/155/653); the weighted result with the distribution of test_eval92 itself is also reported, descriptive only (§0-5).

## 5. Metrics
- Single-talker conditions (C1, C1-R, C0): primary metric "at least one gold value hit" (binding-score-v1, empty interferer set, per recording); also reported: strict intent accuracy, slot P/R/F1, date recall, zero entities, parse failures, counts per c1-error-taxonomy-v1 class.
- Mixture conditions (C3, C5, c6-real-v1): the four binding-score-v1 classes, target-any, "all target values hit", interferer-only (equal-weight and natural angle weighting), pair-flip accuracy; c6-real-v1 additionally reports coverage, the stream-selection error rate / interferer-only / target hits on answered samples (main basis: samples with indistinguishable sector get no credit), and the risk–coverage curve (τ sweep descriptive only; τ = 17° unchanged).
- §11-style diagnostics: entity counts and D_T / D_I hit counts grouped by actual routing correct / incorrect.

## 6. Primary comparison, bootstrap and judgement
- **Direction of the primary comparison: audio LoRA − Whisper + T2-LoRA**, paired on the same audio. Conditions: **C1, C1-R and C5 are jointly primary** (all with target-any; C5 judged on the raw equal-weight basis, as in the devel primary comparison; natural weighting descriptive only). Other comparisons (descriptive only): audio LoRA − original Qwen P2; Whisper + T2-LoRA − Whisper + T2; C0 original T2 / T2-LoRA as the text upper-bound reference; C3, c6-real-v1.
- bootstrap: B = 2000, `np.random.default_rng(2026)`; the primary comparisons report two intervals from the same set of resamples, **98.33%** (0.8333% / 99.1667% quantiles, Bonferroni 3 comparisons) and the uncorrected 95% (2.5% / 97.5%); everything else reports 95% only;
  - single-talker conditions: grouped paired resampling by slurp_id (recordings of the same sentence drawn together);
  - mixture conditions: group level, stratified by angle stratum, the two swapped samples of a group drawn together; for c6-real σ10 the bootstrap is over the mean of the 10 repetitions.
- test has no selection half / report half; all samples are reported together.
- Judgement: each of the three primary comparisons is judged significant only if its **98.33% interval lower bound is > 0**; the 95% intervals are only reported alongside. No correction was applied in the devel stage, and that is not changed retroactively.
- Risk checks (reported only, not a gate): the interferer-only difference and interval of the audio LoRA relative to original Qwen P2 on C3 and the c6-real-v1 main conditions, alongside the devel §6 thresholds; the devel verdict does not change because of test results.
- Whatever the results, no checkpoint, threshold, prompt or route is adjusted after test; anomalies (a surge in parse failures, sample counts not matching, hash mismatches) are only reported.

## 7. Job configuration and GPU estimate
- All GPU jobs: <SLURM_ACCOUNT> / <GPU_PARTITION>, 1 × A100-PCIE-40GB, 4 CPU, 48 GB, **20 minutes** (each job is one route × one shard); SpatialNet front end 1 A100 / 4 CPU / 32 GB / 20 minutes. CPU construction and scoring: <SLURM_ACCOUNT> / <CPU_PARTITION>, 4 CPU, 16 GB, 20 minutes. Submit only with a clean working tree; no automatic retry on failure; reruns only per §0-8.
- Per-item time from devel measurements: P2 / audio LoRA about 1.9–2.6 s, Whisper about 0.32 s, T2 / T2-LoRA about 1.1 s, model loading about 0.5–1.5 minutes / job. Within 20 minutes a P2-type route handles about 400 items / shard.

| Part | Inference calls | Estimated GPU time |
|---|---|---|
| C1 (1,183 × 4 routes, Whisper transcripts shared) | 4,732 | about 2.0–2.5 h |
| C1-R (same as above) | 4,732 | about 2.0–2.5 h |
| C0 (812 sentences × original T2 + T2-LoRA, inference per sentence, expanded per recording) | 1,624 | about 0.5–0.6 h |
| SpatialNet front end (200 groups × 2 = 400 samples) | 400 | < 0.1 h |
| C3 (400 × 2 routes) | 800 | about 0.5–0.6 h |
| C5 both streams (400 × 2 streams × 4 routes) | 3,200 | about 1.4–1.8 h |
| c6-real-v1 | 0 (reuses C5, CPU) | 0 |
| Loading and sharding overhead (about 30–35 jobs) | — | about 0.5–0.9 h |
| **Total** (200 groups) | — | **about 7–9 GPU hours** (cap 10 h, §0-7) |

- Budget: test is separate, capped at **10 GPU hours**, not counted against the 24 h LoRA budget. The estimate for 200 groups is about 7–9 h; if the re-estimate from measurements before execution exceeds 10 h, switch to 100 groups and state so; if the running total is about to exceed 10 h during the run, stop and report.
- Order: CPU construction + sha256 recording and commit → SpatialNet front end → CPU stream selection → single-talker conditions and C3 → C5 both streams → CPU scoring and bootstrap → one-off report.

## Run records

### S0: equivalence check of the test branches against devel (2026-11-12) — **literal criterion not met; stopped before S1**
- Code commit `56731ee` (local, **not pushed**): added test branches to the construction, selection, inference and scoring scripts (`scripts/qwen/split_cfg.py`, default devel; switched only by the environment variable `SLURP_EVAL_SPLIT=test`); added `merge_test_parts.py`, `score_slurp_test_v1.py`, `slurp_test_job.sbatch`, `slurp_test_s0_check.py`.
- Check jobs (<SLURM_ACCOUNT> / debug, 4 CPU, 16 GB, 20 minutes; the <CPU_PARTITION> queue was about 5 hours, so debug was used; not test jobs): construct 22080661 FAILED, select 22080662 FAILED, direct 22080663 PASS, score 22080664 FAILED, dryrun 22080665 PASS. Attribution job 22080668 (old code 5cac35d vs new code 56731ee on the same node) COMPLETED. Results in `outputs/slurp_test_v1/s0/`.
- **What matched**:
  - devel mixture list byte-identical; the 428 16-bit C1-R files byte-identical; mixture manifest identical (except the `dir` path field); direct-path reference arrays and summary identical;
  - `selection.jsonl` and `eval_snapshot_mix.json` byte-identical; C5 eval, c6-real (baseline and LoRA), risk–coverage curves, T2 paired checks and Audio−T2 paired checks all byte-identical to the records; `--ci9833` only adds keys;
  - inference-script dry-run: sha256 of the processed audio of both C5 streams 200/200 identical (including split into two shards), C3 prompts identical, item counts identical;
  - `score_slurp_test_v1.py`: the devel single-talker primary comparisons, descriptive metrics, WER and C3 are all reproduced; smoke-run on devel data reproduces C1 +0.0421 [0.0139, 0.0709], C1-R +0.0748 [0.0376, 0.1132], C5 +0.090 [0.05, 0.13].
- **Four mismatches with the records** (the attribution job confirmed that on the same node **the outputs of old and new code are byte-identical**, so none is caused by the test branches):
  1. Mixture audio: 800/800 FLOAT wav files differ in bytes, but **the samples are bit-identical**; only header bytes 60–62 differ. These three bytes are the PEAK chunk timestamp written by libsndfile. C1-R is PCM_16 and has no PEAK chunk, so it is byte-identical.
  2. The diagnostic field `corr_by_stream` of `per_sample.jsonl` differs in the last 1–2 digits of float64 in 189/200 rows (node / BLAS numerical differences). Stream selection, DOA, polarity and all summaries are identical.
  3. `c6_delta_final/.../eval_c6_delta_spatialnet.json`: the recorded file (11-07) lacks the `route` key. That key was added by commit 4580ad6 on 11-09; everything else is identical. C6δ is not run on test.
  4. `eval_final_lora_checks.py --route-kind p2` raises `KeyError: 'C6delta_+10'` on the current code (including the old code 5cac35d). The C6δ queries added on 11-09 are incompatible with the frozen-consistency assertion of the p2 route, so the 11-07 p2 record cannot be reproduced with the current code. The test branches generate no C6δ queries, so test is not affected.

### S0 verdict: **pass** (project lead 2026-11-12)
- **Classification and handling of the four mismatches**:
  - ① Mixture audio file header: **metadata** (C). In the 800 FLOAT wav files only the write timestamp in the PEAK chunk (from offset 48) differs: rerun `0x6af5d855` (2026-11-12 15:16:37 UTC), record `0x6aeca97b` (2026-11-05 16:06:19 UTC). Samples are bit-identical and all other bytes are identical (`outputs/slurp_test_v1/s0/mix_audio_samplelevel_check.json`, `attrib/attrib.json`). Accepted.
  - ② Selection diagnostic field: **node floating-point difference**, no effect on selection. `corr_by_stream` of `per_sample.jsonl` on debug node <NODE> differs in 189/200 rows, 305 values, in the last 1–2 digits (max difference 8.9e-16); on the login node both old and new code are byte-identical to the record. Selected stream, DOA, polarity, correctness, `selection.jsonl` and summaries are all identical. Accepted.
  - ③ The `route` key of the C6δ output: **metadata** (C). The 11-07 record predates the top-level key `"route": "both"` added by 4580ad6 (11-09); everything else is identical. Accepted; the check script judges "identical after removing that key".
  - ④ Paired checks, p2 route: **fixed and verified**. The C6δ queries added by 4580ad6 have no corresponding condition in the stored c6-real results, causing the `KeyError`. Minimal fix (`eval_final_lora_checks.py`, +7 lines): the p2 route skips queries absent from the stored c6-real results, does not compute them, and records them in the output as `skipped_queries` (C6delta_±10, ±20); other computations are unchanged. After the fix, rerun on devel: after removing the metadata key `route_kind` (added by 4580ad6) and `skipped_queries`, it is **byte-identical** to the 11-07 record. The code version at which the record was generated, 0803c3f, also reproduces the record byte for byte. The test branches generate no C6δ queries anyway.
- **Effect of ② on test**: test runs on compute nodes, so the diagnostic correlation coefficients in `per_sample.jsonl` may differ from the login node in the last digit. They do not enter stream selection, any metric or any verdict.
- **Full devel rerun of the S5 scoring scripts**: with the fixed current code, `slurp_test_s0_check.py score` was run on the login node (`outputs/slurp_test_v1/s0/score_r3_login/check.json`), **17/17 pass**:
  - `eval_final_routes.py` baseline and LoRA: C5, c6-real and risk–coverage curves byte-identical; C6δ judged identical per ③;
  - `eval_final_lora_checks.py` p2 (per ④), t2, audio_vs_t2: values identical; `--ci9833` only adds interval keys;
  - `merge_test_parts.py` split into two shards and merged again: byte-identical to the original file;
  - `score_slurp_test_v1.py`: the devel reproduction check passes; the main flow on devel inputs reproduces C1 +0.0421 [0.0139, 0.0709], C1-R +0.0748 [0.0376, 0.1132], C5 +0.090 [0.05, 0.13].
  - `--natural test_eval92` may only run on test (script assertion) and cannot be rerun on devel; it changes only the weight vector, and the rest of the code path is the same as above.
- The earlier dry-run (22080665) and direct-path references (22080663) passed; for the lists, C1-R and selection results see the previous section.
- Handling of the duplicate commit: the commit 202c6fa produced by a duplicate run has been kept as a backup branch, and master was reset to 8b90d43. The information in it that was verified as correct has been merged into this section and the FSC documents: `per_sample` is reproducible on the login node; `score_v2_r2/` and `score_v2/` are byte-identical; the sha256 of `per_item.jsonl`.
- **After S1 starts, no scoring or construction code is changed.**

### S1: test construction (2026-11-12, <SLURM_ACCOUNT> / <CPU_PARTITION>, 4 CPU, 16 GB, 20 minutes; code b8b41d7, clean working tree)
| Job | Content | Status / time |
|---|---|---|
| 22080676 | mixture list `make_mix_list_v1.py --v2 --split test --n 200` | COMPLETED 0:16 |
| 22080677 | C1-R `reverb_single.py --split test` (test_eval92) | COMPLETED 1:46 |
| 22080730 | mixture audio `mix_slurp_v1.py` (code d72e5c0) | COMPLETED 1:07 |
| 22080731 | direct-path references `direct_refs_mix_v1.py` (sms_wsj environment) | COMPLETED 1:46 |
- **Mixture list** `configs/mix_v1_test_v1.json`, sha256 `26c0fb57e79fa4442526c824cd1d2c79e558af727dc1c11f41e4dea26b02695f`, committed in d72e5c0.
  - Size: 1,396 usable pairs; the integer program for 200 groups was feasible at once (HiGHS Optimal); **no reduction in groups was needed**.
  - 50 groups in each of the four angle strata; the 200 test_eval92 RIR samples are all distinct.
  - 281 recordings used (184/75/22 recordings used 1/2/3 times respectively), 34 speakers; at most 30 groups per speaker; 63 male groups (at least 20 required).
  - T60 0.204–0.500 s; duration difference between the two talkers ≤ 0.96 s.
- **Mixture audio** `data/mix_v1/test_seed2026`: 400 samples, 1,600 wav files, manifest sha256 `a70b7e46…dfdd`.
  - SIR 0 dB (deviation < 5e-15); SNR 19.76–30.12 dB; rir_start 18–43.
  - Maximum peak of the observed signals 1.185, 5 samples with peak ≥ 1; sample points with absolute value ≥ 1: 12 in the observations, 2 in mic0_16k. Files are stored as FLOAT, not clipped; only recorded, no gain adjustment (same as the mix_v1 rules).
- **C1-R** `data/lora_v1/test_rev_eval92`: 1,183 16-bit wav files, matching the set of split_v2 test recordings; the 1,183 test_eval92 samples are all distinct; manifest sha256 `d38900c8…a2da`.
  - **1 clipped file**: `audio-1490703179-headset.flac` (sample 607), peak 1.054 after level adjustment; 2 samples truncated when writing 16-bit.
  - Following the train precedent (§15-5) it is **not regenerated**; recorded as is.
- **Direct-path references** `outputs/slurp_test_v1/direct_refs_mix`: 400 npz files, summary sha256 `bb421901…2dbc`.
  - cv_dev93 check max difference 1.44e-8, pass; early-image recomputation max difference 5.0e-8, pass.
- sha256 of all 3,188 files: `configs/slurp_test_v1_s1_sha256.txt` (sha256 of that file `d6475d2e…d972`).
- **Angle distribution**: 50 groups in each of the four strata of the test mixture set. The natural distribution of test_eval92 itself is 104 / 94 / 207 / 927 (1,332 in total), used only for descriptive weighting; the main results still use the cv_dev93 distribution 78 / 96 / 155 / 653.

### S2: SpatialNet front end and stream selection (2026-11-12; code 9c4bd5b)
| Job | Content | Resources | Status / time |
|---|---|---|---|
| 22080732 | `frontend_snapshot.py --final --mode cv` (cv_dev93 check, 50 groups) | <GPU_PARTITION> 1×A100 / 4 CPU / 32 GB / 20 min | COMPLETED 1:04 |
| 22080733 | `--mode mix` (test 400 samples → `data/mix_v1/separated_final_test/spatialnet_ep100_20261107-223434/`) | same as above | COMPLETED 0:41 |
| 22080734 | `select_final.py` (v3 stream selection + the other stream) | <CPU_PARTITION> / 4 CPU / 16 GB / 20 min | COMPLETED 1:16 |
- **cv check PASS**: errors=[]; direct SI-SDRi 25.933 dB, SI-SDR 20.321 dB; stream selection 0.32 before polarity correction, 1.00 after. Consistent with 22029593 in the devel stage.
- **Separation**: 400 samples × 2 streams, 8 kHz; shape and finiteness checks pass (MIX_DONE 400); manifest sha256 `8301ebdc…1a50`.
- **Stream selection** (descriptive only):
  - mean two-talker direct SI-SDRi on test 18.29 dB, target stream 18.36 dB;
  - stream-selection accuracy: 0.980 overall, 0.997 excluding [0,15);
  - by stratum: [0,15) 0.93, [15,30) 0.99, [30,60) 1.00, [60,180] 1.00; excluding [0,15), the only error is `g162_s2`;
  - polarity flip rate: 1.00 for stream 0, 0.9975 for stream 1.
  - `selection.jsonl` sha256 `ed39744b…5c69`, `other_selection.jsonl` `d40382d6…fe9`, `per_sample.jsonl` `9e4a8b9f…c255`.
- **GPU re-estimate** (per §0-7; using the per-item times measured on devel): P2 1.20 s, audio LoRA 2.05 s, Whisper 0.23 s, T2 1.18 s, T2-LoRA 1.9–2.2 s, C0 1.22 s, C5 P2+W+T2 per stream 2.86 s; loading about 1.6 min per job.
  - S3 + S4: 47 jobs of 20 minutes, inference about 7.4 h, loading about 1.25 h, total about **8.7 GPU hours**; plus 0.03 h for S2, **not above 10 hours; keep 200 groups**.
- **Shards** (each shard estimated ≤ 14 minutes):
  - C1 and C1-R each: P2 2 shards, Whisper+T2 3 shards, audio LoRA 4 shards, T2-LoRA 4 shards;
  - C0: T2 2 shards, T2-LoRA 4 shards;
  - C3: original P2 1 shard, audio LoRA 2 shards;
  - C5 per stream (selected / other): baseline 2 shards, audio LoRA 2 shards, T2-LoRA 2 shards.

### S3 / S4: inference (2026-11-12 11:19–12:23 EST; code 3899ffd; all job logs record the same commit, clean working tree)
- 49 jobs in total: 47 GPU jobs (<SLURM_ACCOUNT> / <GPU_PARTITION>, 1×A100, 4 CPU, 48 GB, 20 minutes) and 2 CPU merge jobs (<CPU_PARTITION>). List in `outputs/slurp_test_v1/runs/jobs.tsv` (job IDs 22080745–22080793).
- Environments: the original Qwen routes use `envs/qwen_omni`; the audio LoRA and T2-LoRA routes use `envs/qwen_omni_lora`.
- **Result**: 49/49 COMPLETED 0:0; **no failures and no reruns**. After each job, sacct and the output were checked: the item count equals the count expected for the shard, `file` has no duplicates, and every line is valid JSON. The longest job was c1_audio_2, at 14:21.

| Condition | Routes (jobs) | Items | Parse status (non-ok) |
|---|---|---|---|
| C1 | original P2 (2) / Whisper+T2 (3) / audio LoRA (4) / T2-LoRA (4) | 1,183 each | P2 bad_json 1; T no_json 3; audio LoRA 0; T2-LoRA 0 |
| C1-R | same as above | 1,183 each | P2 0; T no_json 6; audio LoRA 0; T2-LoRA bad_json 1 |
| C0 | original T2 (2) / T2-LoRA (4) | 1,183 each | T2 no_json 5, bad_json 1; T2-LoRA 0 |
| C3 | original P2 (1) / audio LoRA (2) | 400 each | P2 no_json 1; audio LoRA 0 |
| C5 selected stream | baseline P2+W+T2 (2) / audio LoRA (2) / T2-LoRA (2) | 400 each | P2 no_json 1, T2 no_json 3; audio LoRA 0; T2-LoRA 0 |
| C5 other stream | same as above | 400 each | P2 0, T2 no_json 3; audio LoRA 0; T2-LoRA 0 |
- W (Whisper transcripts) is not in the table: it is not JSON by design, so its parse status is always no_json, as on devel.
- Merge jobs: 22080784 (selected-stream baseline) and 22080791 (other-stream baseline), merged by `merge_test_parts.py` into 400 items each; sha256 recorded in the respective `merge.json`.
- **GPU usage**: S3 + S4 7.803 GPU hours in total; plus 0.029 h for S2, test total **7.83 GPU hours**, within the 10-hour cap (estimate 8.7 h).

### S5: scoring and one-off report (2026-11-12; code 0ce49eb; <SLURM_ACCOUNT> / <CPU_PARTITION>, 4 CPU, 16 GB, 20 minutes)
- **Jobs** (all COMPLETED 0:0, no reruns):
  - merges 22081838–41 (C5 audio LoRA and T2-LoRA, both streams, 400 items each);
  - `eval_final_routes.py`: baseline 22081842, audio LoRA 22081843, and the test_eval92-weighted versions 22081844, 22081845; consistency checks (identical to the C5 scoring when the query direction is θ1) all pass;
  - `eval_final_lora_checks.py`: p2 22081846, t2 22081849, audio_vs_t2 22081850, test_eval92-weighted version of audio_vs_t2 22081851; frozen-metric assertions all pass;
  - report 22081888: `score_slurp_test_v1.py`, input `outputs/slurp_test_v1/score/inputs.json`, output `outputs/slurp_test_v1/score/report/report.json` (sha256 `e457aa3cbdd16a6ec68bfc9204671cda811d2b488ce983c540e074a4d994db8f`).
- All rules were executed in their frozen versions; no rule was changed because of results.

#### Primary comparison (audio LoRA − Whisper + T2-LoRA, target-any, paired; Bonferroni 3 comparisons)
| Condition | n | Audio LoRA | Whisper+T2-LoRA | Difference | 95% interval | **98.33% interval** | Significant (98.33% lower bound > 0) |
|---|---|---|---|---|---|---|---|
| C1 | 1,183 recordings / 812 sentences | 0.8478 | 0.8149 | **+0.0330** | [+0.0110, +0.0545] | **[+0.0064, +0.0595]** | Yes |
| C1-R | 1,183 / 812 | 0.8056 | 0.7506 | **+0.0549** | [+0.0307, +0.0795] | **[+0.0247, +0.0858]** | Yes |
| C5 (raw, equal weight) | 400 samples / 200 groups | 0.7550 | 0.6825 | **+0.0725** | [+0.0349, +0.1125] | **[+0.0250, +0.1200]** | Yes |
- Descriptive only: C5 natural angle weighting (cv_dev93) +0.0575, 95% [−0.0088, +0.1292], 98.33% [−0.0230, +0.1450].

#### Descriptive results (no judgement)
- **Single-talker conditions**:

| Condition / route | target-any | Intent accuracy | Entity F1 (strict) | Value-only F1 | date recall | Parse failures | Zero entities |
|---|---|---|---|---|---|---|---|
| C1 original P2 | 0.567 | 0.696 | 0.324 | 0.506 | 0.288 | 1 | 45 |
| C1 Whisper+T2 | 0.552 | 0.668 | 0.324 | 0.491 | 0.313 | 3 | 41 |
| C1 audio LoRA | 0.848 | 0.909 | 0.743 | 0.779 | 0.850 | 0 | 0 |
| C1 Whisper+T2-LoRA | 0.815 | 0.877 | 0.717 | 0.754 | 0.854 | 0 | 0 |
| C1-R original P2 | 0.529 | 0.672 | 0.304 | 0.476 | 0.279 | 0 | 41 |
| C1-R Whisper+T2 | 0.499 | 0.624 | 0.302 | 0.453 | 0.317 | 6 | 66 |
| C1-R audio LoRA | 0.806 | 0.884 | 0.703 | 0.742 | 0.829 | 0 | 0 |
| C1-R Whisper+T2-LoRA | 0.751 | 0.843 | 0.661 | 0.693 | 0.825 | 1 | 1 |
| C0 original T2 (reference transcripts) | 0.621 | 0.708 | 0.354 | 0.558 | 0.325 | 6 | 36 |
| C0 T2-LoRA (reference transcripts) | 0.868 | 0.904 | 0.766 | 0.805 | 0.888 | 0 | 0 |

  - Whisper WER: C1 0.074, C1-R 0.140.
  - In C0, different recordings of the same sentence give identical outputs; for both routes, 0 sentences differ.
- **Other paired comparisons** (95% intervals):
  - audio LoRA − original P2: C1 +0.281 [0.240, 0.319], C1-R +0.276 [0.237, 0.313], C5 +0.260 [0.205, 0.313];
  - T2-LoRA − Whisper+T2: C1 +0.263 [0.222, 0.302], C1-R +0.252 [0.214, 0.288], C5 +0.258 [0.203, 0.310].
- **C5** (400 samples):

| Route | target-any | Target-only | Interferer-only | Pair flip |
|---|---|---|---|---|
| Original P2 | 0.495 | 0.433 | 0.0025 | 0.250 |
| Whisper+T2 | 0.425 | 0.368 | 0.0025 | 0.165 |
| Audio LoRA | 0.755 | 0.740 | 0.015 | 0.555 |
| Whisper+T2-LoRA | 0.683 | 0.668 | 0.0175 | 0.475 |

  - Whisper WER relative to the target sentence is 0.177.
- **c6-real-v1 main conditions** (natural weighting cv_dev93; in parentheses the equal-weight paired difference Audio − T2-LoRA [95%]):

| Condition / abstention | Coverage | target-any: P2 / WT2 / audio LoRA / T2-LoRA | Interferer-only: P2 / audio LoRA / T2-LoRA |
|---|---|---|---|
| 8 sectors / no abstention | 1.000 | 0.450 / 0.384 / 0.696 / 0.650 (+0.040 [0.008, 0.075]) | 0.026 / 0.050 / 0.049 |
| 8 sectors / abstention A | 0.829 | 0.521 / 0.445 / 0.803 / 0.746 (+0.072 [0.014, 0.131]) | 0.0035 / 0.014 / 0.015 |
| 8 sectors / abstention B | 0.872 | 0.516 / 0.441 / 0.798 / 0.745 (+0.062 [0.011, 0.114]) | 0.0011 / 0.012 / 0.011 |
| σ10 / no abstention | 1.000 | 0.493 / 0.426 / 0.758 / 0.704 (+0.062 [0.026, 0.098]) | 0.024 / 0.048 / 0.045 |
| σ10 / abstention A | 0.855 | 0.520 / 0.451 / 0.804 / 0.750 (+0.065 [0.011, 0.119]) | 0.0022 / 0.011 / 0.012 |

  - The version weighted by the distribution of test_eval92 itself (descriptive only) gives similar values; see `c6_real_test_eval92_descriptive` in `report.json`.
- **Risk checks** (reported only, not a gate; alongside the devel §6 thresholds):
  - C3 (single-channel mixture + direction instruction): interferer-only P2 0.128 → audio LoRA 0.195 (+0.068, above the devel threshold +0.02); target-any 0.118 → 0.240. Fails under the devel §6 rule, the same conclusion as on devel.
  - c6-real naturally weighted interferer-only, audio LoRA − P2: 8 sectors no abstention +0.024, abstention A +0.010, abstention B +0.011; σ10 no abstention +0.024, abstention A +0.009. Against the devel threshold (≤ +0.01), only σ10 abstention A is within the threshold.
  - C5 pair flip 0.250 → 0.555 (not below P2).
  - The interferer-only rate of Whisper+T2-LoRA is similar to that of the audio LoRA (8 sectors no abstention 0.049 vs 0.050), indicating that this increase comes from the LoRAs extracting more entities and is not specific to the audio route.
  - §11 actual-routing diagnostics:
    - With correct routing, interferer-value hits are rare: only 3 in 392 observations in C5; 27 in 3,420 observations in σ10.
    - With incorrect routing, the LoRAs hit interferer values more often than P2: 8 sectors 44 vs 26, σ10 464 vs 270.
- After the above results came out, no checkpoint, threshold, prompt or route was adjusted.

#### test limitations (stated as they are, by decision of the project lead 2026-11-12; numbers from `outputs/slurp_test_v1/score/`, sha256 in the appendix of `docs/slurp_test_report_v1.md`)
- **The devel §6 thresholds are descriptive only on test**; they are not a gate for test and do not change the devel verdict.
- **C3** (single-channel unseparated mixture + direction instruction; `report.json` → `mix.C3`): interferer-only original P2 **0.128 → audio LoRA 0.195** (+0.068; devel threshold +0.02); target-any 0.118 → 0.240. This shows that the audio LoRA does not pick the talker by direction, and on unseparated mixture recordings extracts interferer entities more aggressively.
- **Realistic system path** (c6-real-v1 main conditions, cv_dev93 natural weighting; `checks_p2/checks.json`, `checks_t2/checks.json`): absolute interferer-only values, and the paired differences [95%] of both LoRAs relative to their respective baselines:

| Condition / abstention | Coverage | Original P2 | Audio LoRA | Audio LoRA − P2 | Original Whisper+T2 | Whisper+T2-LoRA | T2-LoRA − Whisper+T2 |
|---|---|---|---|---|---|---|---|
| 8 sectors / no abstention | 1.000 | 0.0260 | 0.0500 | +0.0241 [+0.0106, +0.0421] | 0.0268 | 0.0486 | +0.0219 [+0.0081, +0.0395] |
| 8 sectors / abstention A | 0.829 | 0.0035 | 0.0139 | +0.0104 [0.0000, +0.0292] | 0.0035 | 0.0147 | +0.0111 [0.0000, +0.0292] |
| 8 sectors / abstention B | 0.872 | 0.0011 | 0.0117 | +0.0106 [−0.0011, +0.0291] | 0.0011 | 0.0106 | +0.0094 [−0.0022, +0.0266] |
| σ10 / no abstention | 1.000 | 0.0241 | 0.0479 | +0.0238 [+0.0127, +0.0403] | 0.0216 | 0.0449 | +0.0233 [+0.0122, +0.0388] |
| σ10 / abstention A | 0.855 | 0.0022 | 0.0111 | +0.0089 [+0.0006, +0.0250] | 0.0025 | 0.0122 | +0.0098 [−0.0001, +0.0263] |
  - Abstention B is used only for the sector conditions. Coverage is the naturally weighted value and is the same for all four routes (abstention depends only on stream selection, not on the route).
- **Hit counts split by actual routing correct / incorrect** (no abstention; observations / target-value hits / interferer-value hits):

| Condition | Routing | Original P2 | Audio LoRA | Original Whisper+T2 | Whisper+T2-LoRA |
|---|---|---|---|---|---|
| C5 (query direction = θ1) | correct (392) | 202 / 0 | 327 / 3 | 175 / 0 | 295 / 3 |
| C5 | incorrect (8) | 0 / 1 | 0 / 3 | 0 / 1 | 0 / 4 |
| 8 sectors | correct (337) | 179 / 0 | 285 / 3 | 150 / 0 | 258 / 3 |
| 8 sectors | incorrect (63) | 0 / 26 | 0 / 44 | 0 / 27 | 0 / 42 |
| σ10 (10 repetitions) | correct (3,420) | 1,757 / 0 | 2,848 / 27 | 1,528 / 0 | 2,578 / 27 |
| σ10 | incorrect (580) | 0 / 270 | 0 / 464 | 0 / 239 | 0 / 425 |
  - Each cell is "target-value hits / interferer-value hits"; the number of observations is in parentheses.
- **Notes**:
  - The increase in interferer-only is similar for both LoRAs (8 sectors no abstention: +0.024 vs +0.022; σ10: +0.024 vs +0.023). It comes from the LoRAs extracting entities more aggressively: with correct routing interferer values are almost never hit, while with the wrong stream selected they more readily extract the interferer's entities from that wrong stream. **This is not specific to the audio route.**
  - Abstention (A or B) reduces the increase to about +0.01 (+0.009 to +0.011), but the upper bounds of the intervals are still about +0.025 to +0.029.
- **Limits of the primary comparison**:
  - the 98.33% interval lower bound for C1 is **+0.006**, close to 0;
  - the naturally weighted version of C5 (cv_dev93) is +0.057, 95% [−0.009, +0.129], 98.33% [−0.023, +0.145], **crossing 0, descriptive only**; the main judgement uses the equal-weight (raw) basis (§0-9).

### Reproduction check (2026-11-13, CPU, login node)
- Recomputed from the existing outputs with `scripts/lora/score_slurp_test_v1.py --inputs outputs/slurp_test_v1/score/inputs.json`: the point estimates and 95% and 98.33% intervals of the three primary comparisons all match the report digit for digit, and **report.json is byte-identical** (sha256 `e457aa3c…db8f`).
- The 30 scripts in the pipeline and the local modules they import recursively (64 files in total) are all committed; no untracked code is referenced.
  - The referenced data lists `data/slurp/manifests/split_v2_seed2026/*.csv` are not in the repository (`data/` is in `.gitignore`); they can be rebuilt with `scripts/slurp/make_slurp_split_v2.py`, sha256 in the README.
- The number of git-tracked `outputs/` files is 0; there is no tracked audio or weights (wav, flac, pt, safetensors, npz, etc.). The largest tracked file is `configs/fsc_ext_v1_list.json` (1.27 MB).

### Split-list rebuild check (2026-11-13, CPU; project lead chose option (b))
- **Job**: <CPU_PARTITION> 22092059 (2 CPU, 8 GB, 40 minutes), running `scripts/slurp/make_slurp_split_v2.py` in an exported copy of HEAD 89b1f06. Inputs: `data/slurp/annotations/8eb1654` and `results/slurp/headset_durations.csv` (sha256 `0af9a403…e04e`).
  - The earlier submission 22092056 was cancelled because that duration table (not yet in the repository at the time) was missing from the exported copy.
- **Kept directory**: `<SHARED_ROOT>/tmp_split_rebuild/`. It serves as evidence for the rebuild check, not as a data source:
  - `repo/` is the exported code copy, in which `configs/` and `results/` have been rewritten by the script;
  - `out_manifests/` holds the six rebuilt list files.
- **Results**:
  - the three `*_recordings.csv` are byte-identical to the original files;
  - the three `*_candidate_pairs.csv.gz` are byte-identical after decompression; the file sha256 differs only because of the write timestamp in the gzip header (original 2026-11-05, rebuild 2026-11-13);
  - `configs/slurp_split_v2_seed2026.json` differs only in the path field of the annotation directory (relative / absolute);
  - `results/slurp/split_v2_stats.json` differs only in the 17th significant digit of two total-variation distances and in path fields.
- **Added to the repository**: `data` is a symlink into shared storage, so git cannot track files inside it. The six **original files** (including the 2026-11-05 `.csv.gz`) are therefore added as byte copies under `manifests/slurp_split_v2_seed2026/`; `results/slurp/headset_durations.csv` was removed from `.gitignore` and added. sha256 values (including decompressed contents) are in the README.
