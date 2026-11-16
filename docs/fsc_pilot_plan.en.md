English translation of fsc_pilot_plan.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：fsc_pilot_plan.md

# FSC pilot preparation record (before freeze; no jobs started)

## Data check

The official Fluent.ai description confirms that FSC is an English spoken-command dataset of real recordings: 30,043 utterances, 97 speakers, 16 kHz mono WAV; each CSV record contains the speaker ID, audio path, transcript and the three slots action/object/location. The official split is speaker-disjoint: train 77 speakers/23,132 utterances, valid 10/3,118, test 10/3,793. The licence is CC BY-NC-ND 4.0 for academic research only; the official page also explicitly prohibits commercial use, including training, testing, benchmarking and product development.

The current workspace has no FSC audio, CSV or licence file, only local SLURP data. The official Fluent.ai page and its authorised Google Groups entry were visited; the entry is reachable but currently offers no directly downloadable authorised file; no mirror, scraped link or circumvention of access restrictions was used. "Local accessibility" is therefore still **not obtained**, and no inference job can be submitted. The official licence allows noncommercial academic use but prohibits commercial training, testing and benchmarking; this project's use falls within noncommercial academic scope, and the licence and attribution terms must still be kept. After acquisition, the official archive, CSV, LICENSE, download date and SHA256 must be saved. The existing SLURP train annotation has 11,514 semantic records; the local manifest contains `usrid`, which can be used for speaker auditing. Until the FSC files are in place, the text/semantic/speaker overlap audit between FSC and SLURP can only report its method and verifiable scope, and cannot claim zero overlap.

### Check of the official download address (2026-11-09)

The address `http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz` given in the official Google Groups post returned HTTP 404 for both a HEAD request and a GET with `Range: bytes=0-0`; there was no redirect, and hence no Content-Type or Content-Length. HTTPS on the same port returned a TLS version error, and the default HTTPS port also returned 404. No bytes were downloaded, and no third-party mirror or alternative link was used; the detailed check record is in `configs/fsc_pilot_pending.json`. The current status is `blocked_official_url_404`; work can continue only after Fluent.ai restores that address or provides a new address explicitly authorised by the authors.

### Audit of alternative sources (2026-11-09)

(Public version: the address and acquisition notes of a third-party backup source that were here have been removed. Obtain FSC from the official Fluent.ai channel; for the authorisation status see `DATA_LICENSES.md`.)

(Public version: the address and acquisition notes of a third-party backup source that were here have been removed. Obtain FSC from the official Fluent.ai channel; for the authorisation status see `DATA_LICENSES.md`.)

(Public version: the address and acquisition notes of a third-party backup source that were here have been removed. Obtain FSC from the official Fluent.ai channel; for the authorisation status see `DATA_LICENSES.md`.)

### CPU audit after acquisition (2026-11-09)

After the user confirmed use for noncommercial academic research only, `fluent.tar.gz` was obtained (the public version does not give the third-party source; obtain FSC from the official Fluent.ai channel). The actual file size is 1,552,390,845 bytes, SHA256 `c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b`. Before extraction only the tar members were inspected, and no file inside the archive was executed; the archive contains the official `Fluent Speech Commands Public License.pdf`, whose SHA256 is `2714d08523da55400a57326efb4b42d7ea70b4fccaf3b6abba069a77695fb78d`.

CPU verification passed: train/valid/test have 23,132/3,118/3,793 utterances and 77/10/10 speakers respectively; 30,043 WAV files in total, none missing or corrupt, all 16 kHz, mono, 16-bit PCM. The normalised exact transcript overlap between FSC valid and SLURP train is 349 recordings (25 unique text values); valid has 15 near-duplicate candidates with token 3-gram Jaccard≥0.8, all of which must be flagged as potential text overlap when interpreting results. Speaker IDs across the datasets have no linkable identity, so we can only report "independent namespaces, real-identity overlap not verifiable", and cannot claim full speaker independence. FSC's 31 action/object/location triples and SLURP's 91 intents/18 scenarios are not the same label scheme; semantic overlap is only recorded as requiring a manual mapping audit, and independence cannot be inferred from equal label counts.

The pilot has been changed to 64 frozen valid-only samples, covering 10 speakers (6–7 each) and all 31 FSC intent triples; the manifest SHA256 is `64a695df7de69a93d4e5abea7d13a31b642b8539dd9e9a7301c393a6abe8c173`. The audit and manifest are at `outputs/fsc_pilot/audit.json` and `outputs/fsc_pilot/valid64_manifest.jsonl`; the frozen configuration is at `outputs/fsc_pilot/pilot_config.json`. Neither FSC test nor SLURP final test has been read for any decision; no GPU job has been submitted yet.

### Spatial inputs and smoke status

Following the established RIR/SIR/SpatialNet ep100 configuration, 64 two-talker spatialised directed trials (32 pairs with target/interferer swapped) and 64 single-talker reverberant WAV files were generated. The spatial inputs are 8 kHz, 6-channel, with no clipping; the mixture SIR is 0 dB and the actual SNR is about 24.44–25.49 dB. A CPU pre-check confirmed that the 64 spatial directories, 64 reverberant files, sample rates, channel counts, manifests and hashes are consistent; the FSC valid64 manifest was not changed. Per-sample overlap flags are in `outputs/fsc_pilot/valid64_overlap.jsonl`.

SpatialNet ep100 smoke/full input-check jobs 22057000 (1 smoke item) and 22057001 (64 items) both `COMPLETED/0:0`. Original Qwen smoke 22057002 passed, taking 2:11. Audio LoRA smoke 22057011 failed with ExitCode 1; the log shows that the qwen inference environment lacks the `peft` module. Following the stop-on-failure rule, there was no retry and no environment change, and neither the T2 smoke nor the full pilot was submitted. The current pilot status is **blocked_dependency_missing_peft**; the approved inference environment must be supplemented/confirmed before continuing; the 2 GPU-hours budget was not enlarged because of the failure.

## Overlap audit

After FSC is obtained, the following is always run:

1. Generate an exact-text hash with Unicode lowercasing, punctuation normalisation, and digit and whitespace normalisation; report the exact overlap of FSC train/valid/test with the SLURP **training set**.
2. Screen near-duplicate texts with token 3-gram Jaccard, character 5-gram, and a frozen sentence-transformer cosine threshold (threshold written to the configuration before looking at results); review candidates manually, and do not treat the same intent as text leakage.
3. Treat FSC speaker IDs and SLURP `usrid` as namespaces of different data sources; report speaker overlap only if official metadata provides linkable identities, otherwise mark it "not verifiable" and do not claim no overlap.
4. The audit compares only against SLURP train and does not read or modify SLURP final test. All exclusions are written into the manifest, and the audit script and hashes are saved alongside.

## Frozen semantic scoring

The native FSC target representation is fixed as the triple `{"action": ..., "object": ..., "location": ...}`. The 31 intents are not coarsened directly into a single primary metric:

- `intent_exact`: all three slots equal;
- `intent_family`: auxiliary metric only; the mapping table is fixed as a manually audited action/object table, and the exact result before mapping is reported alongside;
- `slot/entity correctness`: per-slot exact, macro slot accuracy, slot precision/recall;
- `target-any`: any gold slot value of the target speaker is output correctly;
- `speaker-binding`: in two-talker conditions, whether the output slot comes from the target speaker;
- `wrong-speaker-only`: only slot values of the interfering speaker are hit;
- parse failures, abstention coverage and answer risk are reported separately.

If FSC slots cannot be put into one-to-one correspondence with SLURP entity types, compare only at the normalised `concept/value` level, remove unmappable items from the denominator of that metric and report coverage separately; replacing the exact slot result with a loose intent-family mapping is prohibited.

## Pilot design

Select a fixed subset of **64 utterances from the FSC valid split, at least 8 speakers, speaker-disjoint**; sampling seed=2026, stratified by the 31 intents and by speaker; the manifest is frozen before any model prediction. The FSC official test split is held out completely and takes no part in pilot selection, thresholds, models or gate decisions. Each of the four conditions keeps the same set of semantic samples:

- 64 single-talker clean;
- the same 64 as single-talker reverberant with fixed RIRs;
- 64 two-talker spatialised directed trials (32 pairs, target/interferer swapped once per pair);
- DOA ±10°/±20° direction-uncertainty queries on the same two-talker trials.

256 audio/query trials in total; the three frozen routes (original Qwen, Audio LoRA step 440, Whisper+T2-LoRA lr=1e-4 step 660) make 768 inference calls in total. The two-talker conditions reuse the RIRs, sample rate, SIR stratification and route logic of the existing SpatialNet ep100; only the input speech is replaced, and the mixture audio, DOA, route manifest and SHA256 are saved. DOA stream selection is first checked, without looking at model outputs, by comparing the manifest with the actually selected stream; direction queries use the established tau=16° and the frozen C6 rules.

The budget is reserved as one A100, at most 2 GPU-hours (including model loading and log margin); CPU-side transcoding, RIR/manifest work and statistics do not count towards GPU. If any of the following occurs — OOM, input/configuration hash mismatch, sample count not 256, missing rate above 0, duplicate jobs, or a projected excess over 2 GPU-hours — the dependent steps stop immediately and it is reported, with no self-initiated retry or configuration change.

## Pre-prediction freeze list

Data version/licence/hash, speaker-disjoint manifest, exact/near-duplicate thresholds, slot mapping table, audio hashes of the four conditions, RIR/DOA/SIR parameters, prompts, checkpoint provenance, random seeds, scorer version, bootstrap B=2000 seed=2026, abstention threshold and stopping conditions are all written into the pilot config; GPU jobs may be requested only after the checks are complete. This preparation stage started no GPU, did no training and did not read SLURP final test.

### Frozen smoke verification after fixing the dependency (2026-11-09)

Audio LoRA smoke 22057011 failed because it used the base `qwen_omni` environment, which in fact has no `peft`; the failure log and output are kept. After checking earlier successful SLURP jobs, no package was installed or upgraded; instead the verified `<SHARED_ROOT>/envs/qwen_omni_lora` environment was used: the Python executable is its `bin/python`, `peft=0.15.2`, `transformers=4.52.4`, `torch=2.6.0+cu124`, `accelerate=1.7.0`, `safetensors=0.8.0`. The `sys.path` and package locations of the two environments were checked; the original model, checkpoints, prompts, manifests and scoring protocol were not changed.

The fixed Audio LoRA smoke 22057022 used the original resource configuration (1×A100, 4 CPU, 48G, 20 min), `COMPLETED/0:0`, taking 00:01:35; per-sample output is `outputs/fsc_pilot/smoke_audio_fixed.json`. Then Whisper+T2-LoRA smoke 22057024 ran in the same compatible environment with the frozen `step2/lr=1e-4/step660` checkpoint, `COMPLETED/0:0`, taking 00:00:48, output `outputs/fsc_pilot/smoke_t2_fixed.json`. Both logs are kept. Cumulative GPU wall time at this stage (including the failed job) is about 00:05:41, about 0.095 GPU-hours; against the 2 GPU-hours budget about 1.903 GPU-hours remain. The full FSC pilot has not been submitted; awaiting separate approval.

### Pre-submission block of the formal pilot (2026-11-09)

After approval of the full valid64 pilot was received, the pre-submission check confirmed that the frozen inputs are complete in number: valid64 manifest 64, overlap manifest 64, single-talker reverberant 64, spatial_audio_real 64, SpatialNet ep100 outputs 64; the corresponding manifest hashes are `64a695df7de69a93d4e5abea7d13a31b642b8539dd9e9a7301c393a6abe8c173`, `cf80c13c6f34dda1d3db3228de7b78dddc22ed1ad2d0f7822bef5e9ef58b80c7`, `825427d43554cf519bcefb442e2e67f5620029f5e31b39cc8453c7af98e99c9f` and `7dc146cc02b26cd5e4a646bb5d5efc7960c3be3417e2b8a6d08a311cf3bdad23`. There were no FSC jobs in the queue, and the duplicate-job check passed.

However, the FSC-specific DOA selection/selected-stream manifest had not been generated or written to disk; `outputs/fsc_pilot` contains no selection, DOA or route file. Selection results from other experiment directories do not belong to FSC and cannot replace the frozen FSC routing. Since the two-talker and direction-uncertainty conditions depend on that file, work stopped per protocol; no formal pilot GPU job was submitted and no prediction output was generated. Cumulative GPU wall time is still about 00:05:41 (0.095 GPU-hours), with about 1.905 GPU-hours remaining. The FSC route manifest must first be completed and frozen and the CPU input/hash pre-check redone before the pilot can be submitted.

### FSC DOA / selected-stream route manifest (2026-11-09)

An FSC-specific CPU route manifest was produced with the frozen SLURP C5/C6 implementation: zero-lag mic0 polarity correction, PHAT-weighted 360° DOA grid estimation, nearest-angle selection against the frozen direction query, stream 0 on a tie. Selection reads only the FSC 6-channel observation, the two SpatialNet ep100 outputs, the array geometry and `theta_question`; it does not read speaker, transcript, intent or entity. Code provenance: `scripts/sms_wsj/eval_frontend_mix_v1.py` SHA256 `665c9d64228b66a45986ae7bf73ee46b73ca11ff9436fc163b99b795349b7acb`, `eval_frontend_mix_v1_v3.py` SHA256 `046c376c7774aa08a2a89a780cf842d14982b42610aed369076398fdcce79610`, generating script SHA256 `1160cbf5e073319db56df5a8f3225d44f71c95cd51332edb49546c6b15b35171`.

The CPU pre-check generated 64 unique primary route records and 320 direction-uncertainty records (`delta=-20,-10,0,+10,+20°` for each); each record contains sample/pair/swap, target direction, the DOAs of both streams, selected stream, audio path, input/output SHA256, polarity correction, rule and code hash. All 64 inputs and outputs exist, all input/output hashes match, and the pair-swap metadata is complete; in 1 of the 32 pairs the selected streams are complementary (complementarity rate 0.03125); this diagnostic was not used to adjust stream selection.

Reference matching, used only for post-hoc diagnosis, shows a primary selected-stream route accuracy of 0.953125 (61/64); the mean/median DOA error of the selected stream is 0.6243°/0.4866°, and the 16° threshold coverage of the nearest stream is 1.0. The abstention/coverage metadata of the direction-uncertainty records uses the frozen `tau=16°` and was not modified according to FSC semantic outputs. The detailed per-sample files are `outputs/fsc_pilot/route_manifest/route_manifest.jsonl`, `direction_uncertainty_manifest.jsonl` and `diagnostics.jsonl`, summarised in `summary.json`. No GPU job was submitted at this stage; cumulative GPU wall time is still about 0.095 GPU-hours, with about 1.905 GPU-hours remaining.

### Read-only pair-swap / source-level route diagnosis (2026-11-09)

Each FSC pair is made from the same pair of source utterances: swap 1 places A at position 0 and B at position 1 with target=A; swap 2 exchanges the physical positions of the same A/B (A to position 1, B to position 0) with target=B. The target direction query `theta_question` is the same in both, and the target/interferer source identities are swapped. SpatialNet's streams 0/1 are not guaranteed to keep source identity across swaps, so stream indices should not be required to be complementary.

The read-only source matching uses the existing early_A/early_B references for post-hoc permutation matching of the two outputs, without feeding back into the route manifest or stream selection. Source-level results over the 32 pairs: 30 pairs with both correct, 2 pairs with one correct, 0 pairs with both wrong (62/64=0.96875 of the 64). The anomalous pairs are g000 (g000_s1 wrongly selects B, g000_s2 correct) and g011 (g011_s1 correct, g011_s2 wrongly selects A); for both pairs it was confirmed that the target identity swaps A/B, the direction query is the same, and there is no label or pair-construction error. The complementarity rate of the stream index being only 1/32 is an artefact of permutation and cannot be taken as the source-level error rate.

The earlier 61/64 (0.953125) comes from legacy target-only matching of each stream against the target reference; its pair statistics are 29 pairs both correct, 3 pairs one correct, 0 pairs both wrong; the extra g019_s1 is an assignment-definition difference when the correlations of both output streams are weak. Full source permutation matching classifies the g019 pair as both correct; both definitions are kept, and the frozen route manifest and scoring protocol were not changed.

The 0.6243° mean DOA error is the mean absolute circular error over the 64 primary routes of "the DOA of the actually selected stream versus the target direction query"; since the selected stream is itself chosen by nearest DOA, it is numerically equal to the nearest-stream error, and includes the 3 wrong-route samples under the legacy definition (their mean 1.3056°; the mean of the other samples 0.6023°). The 100% at `tau=16°` is only the threshold coverage of the nearest estimated DOA relative to the query direction, not the coverage of the correct speaker.

Diagnostic file `outputs/fsc_pilot/route_manifest/pair_diagnostic.json`, script `scripts/fsc/diagnose_route_pairs.py`. FSC test and SLURP final test were not used, and no GPU job was submitted; the route manifest, algorithm, thresholds and hashes were not changed.

### Block in the first stage of the formal pilot (2026-11-09)

Job 22059111 of the original Qwen stage of the formal pilot failed before the 3rd task (`FAILED/1:0`, 00:02:11). The clean/reverb inputs in the frozen task list are 16 kHz, but the selected SpatialNet `out{k}.wav` of the route manifest is 8 kHz; `qwen_pilot_batch.py` requires 16 kHz under the established model input protocol, so it stopped at the input hash/sample-rate assertion. Only the first 2 tasks were written to `outputs/fsc_pilot/pred_base/predictions.jsonl`; the stage output is incomplete, and the Audio LoRA and Whisper+T2-LoRA stages were not submitted. There was no resampling, no change to the route manifest, task list, model, prompt or scoring protocol, and no automatic retry.

This failure added 00:02:11 of GPU wall time (about 0.0364 GPU-hours); with the earlier ~0.095 GPU-hours, the cumulative total is about 0.131 GPU-hours, with about 1.869 GPU-hours remaining. The failure record and partial output are kept; work continues only after the protocol inconsistency between the 8 kHz SpatialNet selected output and the frozen 16 kHz Qwen input is explicitly resolved.

## Preprocessing fixes and re-freeze (2026-11-11, CPU / login node, no jobs)

### Approval record
- **The FSC valid64 pilot was approved by the project lead on 2026-11-09** (for noncommercial academic research only). This document previously said only "user confirmed" and "approval received", without naming the approver and time; this is recorded here.
- On 2026-11-11 the project lead approved the two preprocessing fixes of this section (FSC scripts only, no change to SLURP code), voiding the old jobs, and untracking the outputs files; the FSC GPU smoke test and trial run are **to be submitted after the SLURP test freeze document is approved**.

### The two inconsistencies found (read-only check, 2026-11-11)
1. **Single-talker reverberation**: the old `prepare_spatial_inputs.py` convolved 16 kHz speech directly with the 8 kHz RIR, then applied `resample_poly(z,2,1)`, wrote it out as 16 kHz and truncated it to the original length (wrong time scale: the content is roughly the first half at half speed); there was no rir_start alignment; level was aligned by peak, not −30 dBFS. Inconsistent with SLURP C1-R.
2. **The P2 routes applied chain8 once too often**: the uncommitted `qwen_pilot_batch.py` called `build(qp, PROMPT_P2, chain8(wav))` in the base / audio_lora routes: clean and reverberant audio went through chain8 (×2 upsampling + level), and the spatial condition went through chain8 twice. The T2 route applied it only once, which is correct.

### Fixes
- New `scripts/fsc/make_reverb16k.py`: **calls `scripts/lora/reverb_single.reverb()` directly** (unchanged; absolute paths make its `SLURP / file` resolve to the FSC files). RIR = the **cv_dev93** sample of the mix_v1 group used by this pilot (`configs/mix_v1_devel100_v2.json` groups[:32], consistent with `spatial_list.json`), position h_{pilot_index mod 2}, mic 0; written as 16-bit PCM (same as reverb_single.py). The old directory was renamed to `outputs/fsc_pilot/reverb16k_void_20261109/` and kept, not deleted. New `reverb16k/manifest.jsonl` sha256 `e22b8da1bac6a0fb63031d36c29b2f6b663efe679b2b0930a1c7881ce911a2d2`; 64 items, no clipping, maximum peak 0.282.
- `prepare_spatial_inputs.py`: the faulty reverberation part was removed (only the generation logic of `spatial_list.json` is kept; not rerun this time).
- New `scripts/fsc/pilot_audio.py`: `prepare()` is the single preprocessing path for each condition; `qwen_pilot_batch.py` now calls it, and the P2 routes no longer call chain8 additionally; the output records `n_chain8`.
- New `scripts/fsc/check_pilot_preprocess.py`: per-condition assertions, result `outputs/fsc_pilot/preprocess_check.json`, **PREPROCESS_CHECK_PASS**.

### Per-condition assertion results (256 items, using the same `prepare()` as inference)

| FSC condition | Corresponding SLURP | Assertion | Result |
|---|---|---|---|
| clean (64) | C1 | 16 kHz; sample-identical to the original FSC file (no level change); chain8 0 times | pass |
| reverb (64) | C1-R | 16 kHz / PCM_16; max difference from recomputation with `reverb_single.reverb()` 3.05e-5 (≤ 1/32768, 16-bit quantisation); active-frame level −30.0000 ± 0.0001 dBFS; chain8 0 times | pass |
| spatial (64) | C5 | input 8 kHz; output sample-identical to a hand-written single `level(resample_poly(sign·x,2,1)[:2n])`; length = 2n; −30 dBFS; polarity multiplied once by `flipped[selected_stream]` of the route manifest (all 64 are −1, consistent with the SLURP ep100 polarity-correction rate of 1.00); chain8 1 time | pass |
| direction_uncertainty (64) | C5 chain (C6δ) | as above; audio paths and selected streams consistent with direction_uncertainty_manifest | pass |

- Reference check: `reverb_single.reverb()` recomputed 3 SLURP devel C1-R files with max difference 3.05e-5 (16-bit quantisation), confirming that what is called is indeed the C1-R chain.
- **Note on length**: the C1-R chain outputs length 2·ceil(n/2). All 428 SLURP devel inputs have even length, so output and input are equally long; in FSC **45 / 64** inputs have odd length, and the reverberant output is 1 sample longer than the clean audio (1/16000 s). This is deterministic behaviour of the chain itself; the chain was not modified.

### Positioning of the direction condition
- The FSC direction condition (fixed offsets ±10°, ±20°) **corresponds to SLURP's C6δ and is a sensitivity analysis only**; it is not real C6 (c6-real-v1) and takes no part in any gate or judgement. **16 items** per offset (assigned by pilot_index mod 4), whereas SLURP C6δ uses all 200 items per offset, so their sample sizes cannot be compared directly. The abstention metadata uses τ = 16° of C6δ ep100 (the τ = 17° of c6-real-v1 does not apply to this condition).

### New frozen task list
- `outputs/fsc_pilot/pilot_tasks.jsonl` (not in the repository): sha256 **`9f9cc65db6ca3b81f1f208a8ee4b9a1d258fcbd265fa9ed923d35af850fd9e36`**, 256 items: clean 64, reverb 64, spatial 64, direction_uncertainty 64 (16 each for −20° / −10° / +10° / +20°). Generated by `scripts/fsc/build_pilot_tasks.py` (logic unchanged).
- The old list is void: sha256 `bdd262745bd7446b49e5e4e801104e7b10fb71839a65bdd818b228b2c002e69b`, copy kept as `pilot_tasks_void_20261109.jsonl`.
- The status and reverberation records in `outputs/fsc_pilot/pilot_config.json` are still the 2026-11-09 version and are out of date; this section takes precedence.

### Voided jobs
| Job | Content | Reason for voiding |
|---|---|---|
| 22057002 | original Qwen smoke | input was the old faulty reverberation `reverb16k/000.wav`, and `qwen_smoke.py` additionally applied `chain8()` to it (one extra ×2 upsampling + level on the P2 route); output "Set the alarm for 7 am tomorrow", unrelated to the original sentence "<FSC utterance>" |
| 22057022 | Audio LoRA smoke | same faulty reverberant input + superfluous chain8; output "play radio" |
| 22057024 | Whisper + T2-LoRA smoke | same faulty reverberant input (no superfluous chain8 on the T2 route); Whisper transcribed " Thank you.", showing that the input audio was already unintelligible |
| 22059111 | original Qwen stage of the formal pilot (FAILED, 2 outputs) | used the old list (with faulty reverberation) and an old uncommitted version of `qwen_pilot_batch.py`; stopped at the 8 kHz assertion; output `pred_base/` incomplete |
- The GPU time of these jobs (about 0.131 GPU hours) still counts towards the FSC budget of 2 GPU hours; about 1.869 GPU hours remain.

### outputs files untracked
- `outputs/` is in `.gitignore`, but the following 11 files had been force-added to tracking. On 2026-11-11 they were untracked with `git rm --cached`; the files on disk are unchanged:

| File | Bytes | sha256 |
|---|---:|---|
| outputs/fsc_pilot/audit.json | 9813 | `cc619fae7e039f09dce2ba55fc6e587c7719b5dd70c22d86b1d6a028dba36643` |
| outputs/fsc_pilot/pilot_config.json | 5745 | `031276d1e8ddba7cdc8afd2e7fa0604f9b0e877a0418ee1c1e588b13cf7f7d3c` |
| outputs/fsc_pilot/pred_base/predictions.jsonl | 1177 | `a343f2e84f81f81d359d3bdfcd05a13fb7838aa7456bd0e85338514984c13116` |
| outputs/fsc_pilot/route_manifest/diagnostics.jsonl | 28741 | `03e789efd483d21694980a8cdf116cf3e359fcdc869a3475cb17596c3d75a652` |
| outputs/fsc_pilot/route_manifest/direction_uncertainty_manifest.jsonl | 359557 | `7d8ed8364e6e08435eec5c4685cfa61363c6d20adf00fd1ff851ed1351474aef` |
| outputs/fsc_pilot/route_manifest/pair_diagnostic.json | 66128 | `e1e57a6cf13145e58a3bf824cf15fd7d67691d2edd2d4dc2b678989c19e4d838` |
| outputs/fsc_pilot/route_manifest/route_manifest.jsonl | 81052 | `de361ee54ec8772b89237b3579388e900e348ab5d168cbcc74d339d56402522e` |
| outputs/fsc_pilot/route_manifest/summary.json | 2376 | `91d86e712c058f366a10121bf9cdf57f520fd752865ec588a920c9141acd5327` |
| outputs/fsc_pilot/spatial_list.json | 41532 | `e114f68eb4bea7dc99aa42be25baab9173d9a5b312a433df74f2baeaee0fa903` |
| outputs/fsc_pilot/valid64_manifest.jsonl | 22861 | `64a695df7de69a93d4e5abea7d13a31b642b8539dd9e9a7301c393a6abe8c173` |
| outputs/fsc_pilot/valid64_overlap.jsonl | 34997 | `cf80c13c6f34dda1d3db3228de7b78dddc22ed1ad2d0f7822bef5e9ef58b80c7` |

- Newly generated files: `reverb16k/manifest.jsonl` `e22b8da1…a2d2`; for the sha256 of `preprocess_check.json` see the record at commit time (recomputable with `check_pilot_preprocess.py`).
- `scripts/fsc/qwen_smoke.py` still contains the superfluous chain8 and was not changed this time; future smoke tests use `qwen_pilot_batch.py --limit N` (going through the same `prepare()`) instead of `qwen_smoke.py`.

### Addendum (2026-11-11, at the project lead's request)
- **Reverberant output 1 sample longer** (recorded explicitly again): the reverb condition is produced with the SLURP C1-R chain (`reverb_single.reverb()`), with output length 2·ceil(n/2). In FSC valid64, **45 / 64** original recordings have odd length, and their reverberant output is 1 sample longer than the clean audio (1/16000 s, at the end); the 19 even-length inputs give equal length. All SLURP devel inputs have even length, so this never occurred on SLURP. This is deterministic behaviour of the chain itself; the chain was not modified and nothing was truncated; `check_pilot_preprocess.py` asserts this length rule (`n_len_plus1_odd_input = 45`).
- The beginning of `scripts/fsc/qwen_smoke.py` now marks it as deprecated and exits immediately; smoke tests can only use `qwen_pilot_batch.py --limit N`.
- Decision of the project lead on 2026-11-11: **no smoke test**; submit the 256 × 3 formal run directly (one job each for base, audio_lora, t2_lora; <SLURM_ACCOUNT> / <GPU_PARTITION>, 1 × A100, 4 CPU, 48 GB, 20 minutes; `envs/qwen_omni_lora`). Submission script `scripts/fsc/qwen_pilot_batch.sbatch`, output `outputs/fsc_pilot/pred_v2_<route>_<jobid>/`; the old `pred_base/` (22059111) is kept untouched and void. Run results and check records are in the next section.

### Formal run v2 (2026-11-11, 256 × 3, no smoke)
The working tree was clean before submission; the logs of all three jobs record `GIT_COMMIT 28b1337`, the `envs/qwen_omni_lora` python, and tasks sha256 check OK.

| Route | Job | sacct | Time | Items / per condition | raw_output parsable JSON (with intent+entities) |
|---|---|---|---|---|---|
| base | 22076114 | COMPLETED 0:0 | 00:08:07 | 256 / 64×4 | 256 / 256 |
| audio_lora | 22076115 | COMPLETED 0:0 | 00:11:54 | 256 / 64×4 | 256 / 256 |
| t2_lora | 22076116 | COMPLETED 0:0 | 00:12:14 | 256 / 64×4 | 256 / 256 |

- Item-by-item check: index, condition, pilot_index and audio_sha256 match the task list; route, checkpoint and prompt are correct; n_chain8 is 0 for clean/reverb and 1 for spatial/direction; only t2_lora has asr.
- Manual spot check: 2 items per condition (`random.Random(2026)`), drawing clean #15/#40, reverb #13/#28, spatial #53/#62, direction #30(+10°)/#56(−20°), 24 items over the three routes. Both the model transcripts and the Whisper transcripts match the original sentences. The only deviation is clean #40, where Whisper transcribed "Turn down" as "Countdown"; this is an ASR error, not a pipeline problem.
- GPU: 0.54 GPU h this time; FSC cumulative about 0.67 / 2 GPU h.
- Not yet scored: the mapping from FSC labels to SLURP intents/slots has to be done separately as planned.
- Output sha256 (`outputs/fsc_pilot/`, not in the repository): `pred_v2_base_22076114/predictions.jsonl` 825ddd5b…c9be; `pred_v2_audio_lora_22076115/predictions.jsonl` 37b5c7b7…b28f; `pred_v2_t2_lora_22076116/predictions.jsonl` 63b09d59…d4d0.

### Scoring (2026-11-12, CPU, `scripts/fsc/score_pilot.py`; under the rules frozen before the run, with no change to or addition of mappings)
- Inputs: the `predictions.jsonl` of the three v2 formal runs (sha256 in the previous section); task list sha256 `9f9cc65d…9e36`; mapping file `configs/fsc_semantic_mapping.json` (frozen 2026-11-09, unchanged). Outputs `outputs/fsc_pilot/score_v2/fsc_pilot_scores.json` (sha256 `c607d2e6eb091f98a91fa23f24188a8fa54b0ec37482a8287668adba4ac462cf`) and `per_item.jsonl` (sha256 `9b092d7307fb1bafac94121dc700209b50f67a7d132e11b33bdb4c4e9c39d231`). Rerunning the same script into `score_v2_r2/` gives byte-identical files.
- **Mapping coverage**: the frozen file contains only rules; it has **no** audited SLURP output → FSC triple mapping table, and no correspondence between FSC slots and SLURP entity types. Therefore:
  - `intent_exact` (all three slots correct) **cannot be computed** (coverage 0/256), and `intent_family` was not computed; no loose mapping was substituted.
  - All other metrics are computed at the "normalised value level" allowed by the frozen rules: gold slot values (excluding `none`, `normalize_v2`) are matched exactly as whole values against model output entity values, ignoring types.
  - action values (activate / deactivate / increase / decrease / bring / change language) are intents rather than entities in the SLURP format and almost never appear as entity values, so action recall is close to 0. This is caused by the missing mapping and **is not a conclusion about the models**.
  - Output coverage (the proportion of model output entity values equal to some FSC slot value): clean 0.68–0.83, spatial 0.59–0.75, see the "Coverage" column below; other output values (e.g. `washroom lights`, `english`) score nothing under the frozen rules.
- Binding in the two-talker conditions uses binding-score-v1 `judge` (D_T = target − interferer); in 2/64 items the two talkers' slot values do not differ, and these are removed from the binding metrics. "Target-only" requires all of D_T to be hit, and D_T contains the action, so this category is generally low; "interferer-only" is wrong-speaker-only.
- Direction-uncertainty condition: answering is determined by `within_tau` of the route manifest (frozen C6 τ=16°); coverage 0.672 (−20° 0.25, −10° 1.00, +10° 1.00, +20° 0.44).
- Confidence intervals are descriptive only: B=2000, seed 2026; single-talker conditions resampled by utterance, two-talker conditions by pair.

| Route | Condition | target-any [95%] | object / location recall | value P / R | Coverage | Interferer-only | Transcript WER | Zero entities |
|---|---|---|---|---|---|---|---|---|
| Original Qwen | clean | 0.609 [0.484, 0.719] | 0.468 / 0.417 | 0.684 / 0.260 | 0.684 | — | 0.047 | 8 |
| Original Qwen | reverb | 0.562 [0.438, 0.688] | 0.435 / 0.375 | 0.643 / 0.240 | 0.661 | — | 0.101 | 9 |
| Original Qwen | spatial | 0.500 [0.359, 0.625] | 0.371 / 0.375 | 0.571 / 0.213 | 0.589 | 0.016 | 0.141 | 9 |
| Original Qwen | direction | 0.328 [0.203, 0.469] | 0.274 / 0.167 | 0.375 / 0.140 | 0.661 | 0.258 | 0.418 | 8 |
| Audio LoRA | clean | 0.781 [0.687, 0.875] | 0.419 / 0.917 | 0.758 / 0.333 | 0.758 | — | 0.027 | 0 |
| Audio LoRA | reverb | 0.766 [0.656, 0.875] | 0.403 / 0.917 | 0.742 / 0.327 | 0.758 | — | 0.067 | 0 |
| Audio LoRA | spatial | 0.641 [0.500, 0.766] | 0.323 / 0.833 | 0.612 / 0.273 | 0.642 | 0.016 | 0.104 | 0 |
| Audio LoRA | direction | 0.453 [0.312, 0.594] | 0.258 / 0.500 | 0.433 / 0.193 | 0.672 | 0.258 | 0.384 | 0 |
| Whisper+T2-LoRA | clean | 0.734 [0.625, 0.844] | 0.419 / 0.875 | 0.797 / 0.313 | 0.831 | — | 0.057 (Whisper 0.088) | 6 |
| Whisper+T2-LoRA | reverb | 0.688 [0.578, 0.797] | 0.387 / 0.833 | 0.721 / 0.293 | 0.770 | — | 0.165 (0.165) | 4 |
| Whisper+T2-LoRA | spatial | 0.641 [0.516, 0.766] | 0.355 / 0.792 | 0.683 / 0.273 | 0.750 | 0.016 | 0.108 (0.138) | 4 |
| Whisper+T2-LoRA | direction | 0.406 [0.281, 0.547] | 0.242 / 0.458 | 0.433 / 0.173 | 0.750 | 0.274 | 0.394 (0.428) | 4 |

- Parse failures are 0 for all three routes, and abstentions are 0. action recall: Audio LoRA 0.016–0.031, the other routes 0.
- The direction condition counts answered samples only: target-any original Qwen 0.256, Audio LoRA 0.419, T2-LoRA 0.372; interferer-only 0.357 / 0.357 / 0.381.
- Descriptive paired differences (Audio LoRA − Whisper+T2-LoRA, target-any): clean +0.047 [−0.031, +0.125], reverb +0.078 [0.000, +0.172], spatial 0.000 [−0.078, +0.094], direction +0.047 [−0.016, +0.109]. All intervals contain or touch 0. The pilot has only 64 items per condition; **descriptive only, no verdict**.
