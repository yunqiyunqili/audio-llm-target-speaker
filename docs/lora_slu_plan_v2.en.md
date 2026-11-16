English translation of lora_slu_plan_v2.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：lora_slu_plan_v2.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# LoRA fine-tuning plan v2 (lora-slu-v2; **frozen 2026-11-06**; written before the run and not changed after the results were seen)

> - Rewritten from `docs/lora_slu_plan_draft.md` (v1 draft, kept unchanged) according to the project lead's comments of 2026-11-06.
> - Frozen version: saved separately from `docs/lora_slu_plan_v2_draft.md` after adding the decisions of 2026-11-06. At the freeze **no training had been done yet**; the environment had been set up (§10). The times, GPU memory and disk in §8 are still **estimates**; the measurements of the step-0 pilot test take precedence.
> - Training uses only split_v2 **train**; selection uses only **devel**; **test is not touched** until the test rules of §9 are separately frozen.
> - mix_v1, the P2/T2 prompts, slu-score-v2, binding-score-v1, c1-error-taxonomy-v1, the v3 stream-selection rule, existing results and verdicts are not changed; training jobs are not touched.

## 1. Basis (existing results)
- C1/P2 0.694, Whisper + T2 0.653, C0 (reference transcripts + T2) 0.782, all on 124 devel recordings, per recording (`c0_plan_v1.md`).
- c1-error-taxonomy-v1: under all three inputs, omissions account for about 48–59% of missed entities (15 of the 27 omissions in C0 are date), with about 18 boundary / type mismatches; removing "listening" only eliminates the "misheard" class.
- The gain from LoRA may therefore come mainly from learning SLURP's entity annotation spans, which applies equally to the text route. The primary comparison must be **the paired difference between the audio LoRA and Whisper + T2-LoRA on the same audio**.

## 2. Steps and order
| Step | Content | Input → output | Evaluation | Status |
|---|---|---|---|---|
| 0 | **Text LoRA** | train reference transcripts + frozen T2 → JSON in the T2 format | devel reference transcripts (C0 basis) | first |
| 1 | **Audio LoRA L1** | train clean recordings + single-talker full reverberation mic 0, frozen P2 → JSON in the P2 format | all conditions of §5 | after step 0 |
| 2 | **Whisper + T2-LoRA** | Whisper large-v3 (`06f233fe…`) transcribes **the same train audio as step 1**, + frozen T2 → JSON in the same format; same training settings as step 1 | same audio and conditions as step 1 | paired with step 1 |
| L2 | audio LoRA plus direct-path references | — | — | **optional**, needs separate approval |
| L3 | add the separated target stream | — | — | **postponed** until after the completed SpatialNet checkpoint |
- Step 0 is **used only to measure the gain from "learning SLURP's annotation spans"** (without listening); it is not evidence for "whether the audio LLM is better".
- All three steps add LoRA only to the thinker's language model: q, k, v, o (**restricted by the regex `model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)` to the 28 layers of the thinker language model, 112 modules in total**; the audio encoder `audio_tower` also has 96 q/k/v_proj with the same names, so simple name matching cannot be used), r = 16, alpha = 32, dropout 0.05; audio encoder frozen, talker disabled; bf16, gradient checkpointing, AdamW, learning-rate candidates {5e-5, 1e-4, 2e-4} (at most 3 settings for step 0; **2 settings each for steps 1 and 2**, {5e-5, 1e-4}, the same for both), cosine decay, 3% warmup, effective batch 16, at most 3 epochs, seed 2026. All three steps use the same hyperparameter candidates and the same checkpoint interval.
- Training target: JSON generated from SLURP gold labels (transcript = the original SLURP sentence, intent, entities), with loss on the answer tokens only. **The transcript field of step 2 also uses the original SLURP text**, the same as the audio LoRA (decision of 2026-11-06).

## 3. Data
### 3.1 train (checked; see v1 draft §2.1)
- 3,513 recordings, 66 speakers, 2,363 distinct sentences, 2.90 hours; the intersection of sentences and slurp_ids with devel / test is 0.
- Step 0: **2,363 texts** after deduplication by slurp_id (multiple recordings of the same sentence have the same text).
- Steps 1 and 2: 3,513 clean recordings + 3,513 single-talker full reverberations (RIRs from SMS-WSJ **train_si284**, 8 kHz convolution → 16 kHz, −30 dBFS, same as the C5 chain), **7,026 recordings** in total. The reverberant audio needs to be newly generated, about 0.34 GB (16-bit), stored in `<SHARED_ROOT>/data/lora_v1/`.

### 3.2 Selection half and report half of devel
- **Single-talker conditions now use all 428 devel recordings** (382 sentences, 11 speakers, 590 gold entities, 128 of them date; checked 2026-11-06).
- Split method (by recording, with recordings of the same sentence in the same half): sort the 382 devel slurp_ids ascending, `random.Random(2026).shuffle`; all recordings of the first 191 slurp_ids form the **selection half**, the rest the **report half**. Result checked under this rule: selection half **214** recordings, report half **214**; of the 124 mix_v1 recordings, 63 / 61 fall in each.
- Mixture conditions (C4-direct, C5, c6-real-v1) keep the group split of c6-delta-v1 (100 group numbers, `random.Random(2026).shuffle`, first 50 groups selection, last 50 groups report). The two splits are independent; the same recording may be in the selection half of the single-talker conditions but in the report half of the mixture conditions; they are reported separately in evaluation and not mixed.
- Text conditions (C0 basis) are split by the same slurp_ids: selection half 191 sentences, report half 191 sentences; metrics are reported both per sentence and per recording (428 recordings, repeated sentences weighted by recording).

## 4. Primary comparison and the criterion for "improvement"
- **Primary comparison**: the paired difference between the step-1 audio LoRA and the step-2 Whisper + T2-LoRA **on the same audio** (audio LoRA − Whisper + T2-LoRA).
- Other comparisons: audio LoRA − P2; Whisper + T2-LoRA − Whisper + T2; text LoRA − C0 (step 0).
- **Paired bootstrap**: B = 2000, `np.random.default_rng(2026)`; within one resample both routes use the same samples; take the 2.5% / 97.5% quantiles of the difference.
  - Single-talker conditions: **recording level**, grouped resampling by slurp_id (recordings of the same sentence drawn together);
  - Mixture conditions: **group level**, stratified by angle stratum (same as c6-real-v1).
- **"Improvement"**: the **lower bound** of the 95% interval of the difference **> 0**. The selection half is used to choose hyperparameters and checkpoints (best point estimate) and to decide whether to proceed to the next step; "improvement" for external reporting is judged only on the **report half**.
- Primary metric: binding-score-v1 target-any; for mixture conditions also interferer-only (natural angle weighting).

## 5. Evaluation conditions and diagnostic metrics
| Condition | Samples | Used for |
|---|---|---|
| C1 (clean) | 428 devel recordings | steps 1, 2; degradation check |
| C1-R (single-talker full reverberation, cv_dev93 RIRs) | 428 | steps 1, 2 (matching the training distribution) |
| C0 (reference transcripts) | 382 sentences / 428 recordings | step 0; degradation check |
| C4-direct | 200 samples | steps 1, 2 |
| C5 (selected stream of the SpatialNet snapshot) | 200 | steps 1, 2 (snapshot diagnosis) |
| c6-real-v1 main conditions (8 sectors, σ = 10°; no abstention, abstention A, abstention B) | 200, inference on both streams | steps 1, 2; direction-related degradation check |
| C3 (single-channel mixture + direction instruction) | 200 | degradation check of step 1 only |
- **New diagnostic metrics**:
  - **date entity recall**: the proportion of gold entities of type date that are hit under binding-score-v1 value matching (normalize_v2, type ignored);
  - **change in the counts of each c1-error-taxonomy-v1 class** (a–f, rules unchanged; e / f for C0 / text conditions judged with reference transcripts, the Whisper route with Whisper transcripts).
- **Baselines that need to be added** (existing baselines cover only 124 recordings): P2, Whisper + T2 and C0 on the 428 devel recordings; devel audio for C1-R (cv_dev93 RIRs, about 0.04 GB). All of these need separately approved jobs.

## 6. Degradation checks (all items of the v1 draft kept; single-talker conditions changed to the selection half of the 428)
1. C1 target-any no more than 0.03 below P2; intent accuracy no more than 0.03 lower; parse failures no more than 2%.
2. C4-direct not below P2 − 0.03.
3. Under the c6-real-v1 main conditions, weighted interferer-only no more than 0.01 above P2 (checked for no abstention, abstention A and abstention B); on C3, interferer-only no more than 0.02 higher and target-any not below P2 − 0.03; C5 pair-flip accuracy not below P2.
4. Interferer-only of the other C5 stream is only reported.
5. The text LoRA (step 0) and Whisper + T2-LoRA (step 2) are likewise checked for no degradation on C0 / C1 (against their respective non-fine-tuned baselines).
- Degradation checks use point estimates (empirical thresholds), separate from the bootstrap criterion for "improvement".

## 7. Stopping conditions
1. devel / test sentences, recordings or speakers appear in the training data, or cv_dev93 / test_eval92 RIRs are used → stop immediately.
2. Pilot test: peak GPU memory at batch 1 > 38 GB, or the budget at the measured speed would exceed §8 → stop and report.
3. Training loss NaN or diverging → the job stops; only the reason is reported; no resubmission with a changed configuration.
4. **Step 0**: if the 95% interval lower bound of the paired difference text LoRA − C0 on the selection half is ≤ 0 → **stop; do not proceed to steps 1 and 2**; report as "LoRA gives no clear help" (decision of 2026-11-06). **Exception**: if the training loss did not decrease, or the output format broke (e.g. a clear increase in parse failures), report the reason first; this does not count as no gain.
5. **Step 1**: if the interval lower bound of audio LoRA − P2 (C1) on the selection half is ≤ 0, or §6 is not passed → stop, with no follow-up beyond step 2 (no L2); step 2 is still done, to complete the paired comparison.
6. Hyperparameters: at most 3 settings for step 0 and 2 settings each for steps 1 and 2; total GPU time reaching the §8 cap → stop.

## 8. Compute and budget (estimates, not measured)
- Resources: 1 A100-PCIE-40GB (<GPU_PARTITION>, account <SLURM_ACCOUNT>), 4 CPU, 48–64 GB memory.
- **Pilot test, 30 minutes**: measure time per step and peak GPU memory over 200 steps; afterwards all times are re-estimated from measurements and reported.
- **Each job within 30 minutes, resumed in segments**: before each segment ends (about 25 minutes), save the adapter, optimizer state, learning-rate schedule, data order and RNG state; the next segment continues from the save point, and the log records the start and end step of each segment. Resume jobs are submitted segment by segment, with no automatic resubmission (unless separately approved).
- **Total GPU time cap 20 hours**, allocated (estimates):

| Item | Estimate | Basis |
|---|---|---|
| Pilot test | 0.5 h | — |
| New baselines on 428 (P2, Whisper, T2, C0) + P2 / Whisper + T2 on C1-R | about 1 h | P2 about 1.9 s per item, Whisper about 0.32 s, T2 / C0 about 1.1 s |
| **Step 0** training | 2,363 × 3 epochs ≈ 7,100 forward/backward passes; 0.3–0.6 s each (input about 600 tokens, of which the T2 prompt is about 560) → **0.6–1.2 h / setting**; 3 settings ≈ 2–3.5 h | C0 dry-run input length 563–567 tokens |
| Step 0 evaluation | per checkpoint 428 items × about 1.1 s ≈ 8 minutes; 2 per setting → about 0.8 h | C0 measured 1.13 s / item |
| Step 1 training + evaluation | training 7,026 × 3 epochs, 2–5 h per setting; evaluation about 1 h per checkpoint; **at most 2 settings** → about 6–9 h | — |
| **Step 2** Whisper transcription of train | 7,026 items × about 0.32 s ≈ **37 minutes** (2 segment jobs) | W_C1 measured 124 items in 39.2 s |
| Step 2 training + evaluation | text input, 7,026 × 3 epochs ≈ 2.1 × ten thousand passes, 0.3–0.6 s each → 1.8–3.5 h; evaluation about 0.5 h / checkpoint → about 3–4.5 h | — |
| Total | about 14–20 h | if above the cap, stop per §7 item 6 |

- **Disk (estimates)**:
  - Step 0: about 20–80 MB per adapter, at most 10 kept, ≤ 0.8 GB; output jsonl about 10 MB.
  - Step 2: Whisper transcript jsonl about 3 MB; no new audio (uses the 7,026 recordings of step 1); adapters ≤ 0.8 GB.
  - Step 1: reverberant audio about 0.34 GB; devel C1-R about 0.04 GB; adapters ≤ 0.8 GB.
  - New environment: see §10, at most about 6 GB.
  - `<SHARED_STORAGE>` is currently 87% used, with about 136 GB remaining.

## 9. test rules (frozen in a separate document before the run)
- Run **once**, only on the **final frozen configuration**.
- Conditions limited to: C1, C0, C5 with the **completed** SpatialNet **checkpoint**, and the c6-real-v1 main conditions (8 sectors, σ = 10°).
- Before the run, a separate test-rules document is written and committed as frozen, stating the configuration, routes, metrics, bootstrap and judgement; after the test results come out there is no further tuning.

## 10. Environment (set up 2026-11-06)
- The existing `envs/qwen_omni` was **not changed** (`pip freeze` identical before and after installation).
- PyPI metadata (queried 2026-11-06): peft 0.15.2, released 2025-04-15, `py3-none-any` wheel 411,051 bytes, sha256 `0dfc942b…93d0a0`; requires Python ≥ 3.9; dependencies numpy ≥ 1.17, packaging ≥ 20.0, psutil, pyyaml, torch ≥ 1.13.0, transformers (no version limit), tqdm, accelerate ≥ 0.21.0, safetensors, huggingface_hub ≥ 0.25.0. The existing environment satisfies all of them (numpy 2.2.6, packaging 26.3, psutil 7.2.2, pyyaml 6.0.3, torch 2.6.0, transformers 4.52.4, tqdm 4.70.1, accelerate 1.7.0, safetensors 0.8.0, huggingface_hub 0.36.2). The metadata places no version limit on transformers; compatibility is confirmed by the actual test below.
- Download: the wheel is stored in `<SHARED_ROOT>/downloads/peft_0.15.2/`, sha256 matching PyPI.
- Setup (srun 22023507, <CPU_PARTITION>, <NODE>): `conda create --clone envs/qwen_omni -p envs/qwen_omni_lora` (about 7 minutes), then `pip install --no-deps --no-index <wheel>`. The new environment is 5.8 GB; compared with the original environment, `pip freeze` only adds peft; `pip check` reports no conflicts.
- Test (srun 22023514, CPU, `scripts/lora/peft_env_check.py`): importing torch 2.6.0+cu124 / transformers 4.52.4 / accelerate 1.7.0 / peft 0.15.2 succeeds; adding LoRA (q/k/v/o, r 16) to a small randomly initialised Qwen2: initial outputs identical to the original model, only LoRA parameters trainable, gradients present in backpropagation; on the Qwen2.5-Omni thinker configuration (meta device, no weights loaded), 112 q/k/v/o_proj were found in the 28 language-model layers, with another 96 modules of the same names in the audio encoder (hence the regex restriction in §2).
- Estimated number of LoRA parameters (r = 16, hidden 3584, k/v dimension 512): about 36 × ten thousand per layer, about 1,009 × ten thousand over 28 layers.

## 10b. GPU priority (decision of 2026-11-06)
- Reruns after SpatialNet training completes take **priority over** LoRA; when the two conflict, SpatialNet runs first (when SpatialNet completes, report first and wait for approval).

## 11. Decision record (2026-11-06, project lead)
1. Approved creating `envs/qwen_omni_lora` and installing peft 0.15.2 (done, see §10).
2. Step 1 tries 2 hyperparameter settings.
3. The transcript field in the step-2 training target uses the original SLURP text.
4. Approved the 428-recording baselines, in order: run C0 (reference transcripts + T2) on 428 now; before step 1, run P2 on 428, Whisper + T2 on 428, and the single-talker reverberant devel audio (generated on CPU). For each job, report the configuration first and submit with a clean working tree.
5. The thresholds of §6 and §7 are kept.
6. The step-0 stopping condition is changed to the wording of §7 item 4.
7. Frozen as `docs/lora_slu_plan_v2.md`; after the freeze, first submit the step-0 pilot test (30 minutes), report the measured speed and GPU memory, and start formal training only after approval.
8. GPU priority: see §10b.

---

## Revision 1 (2026-11-06 16:10 EST; **written before any training results**: at this point only the pilot test (speed, GPU memory) and the C0-428 baseline had been run, with no formal training)
1. **Primary metric for the 428 single-talker recordings**: **"at least one gold value hit"** (value matching without an interferer: binding-score-v1 normalize_v2 value-multiset matching, D_T = all gold values of the recording, D_I empty). Used for selection and gating.
2. **Step-0 training configuration**: three learning rates 5e-5, 1e-4, 2e-4; micro-batch 4 with gradient accumulation to an effective batch of 16; 3 epochs; seed 2026; a checkpoint every 0.5 epoch; each setting in 2 segments, each within 30 minutes.
3. **Step-0 gate**: on the **selection half**, the paired bootstrap difference of the best checkpoint relative to the C0-428 baseline (job 22023517) (**per sentence**, i.e. grouped resampling by slurp_id, B = 2000, seed 2026) must have a 95% interval lower bound > 0 to pass; otherwise stop per §7 item 4 (including its exception).
4. **Also reported** (selection half, report half and all, separately): paired target-any on the 124 mix_v1 recordings (compared with the 0.782 of C0 v1), date recall, counts of each c1-error-taxonomy-v1 class.
5. **The training log** additionally records the loss on the entity tokens (the tokens after `"entities"` in the target JSON), **for observation only**; the training target is not changed.
6. **Resume verification**: after the first segment of the first setting, verify at the start of the second segment that the global step, learning rate, optimizer state and data order after resuming match an uninterrupted run; **if verification fails, stop and report; do not retry with shorter segments** (whether to request a 60-minute limit is decided separately).
