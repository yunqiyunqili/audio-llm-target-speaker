English translation of plan_update_v2.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：plan_update_v2.md

# Plan update v2 (2026-11-06)

> **Scope (all numbers are subject to these limits)**
> - All results come **only from the 100 devel groups of mix_v1** (200 swapped samples; single-talker conditions use 124 recordings or their mapping), each condition run only once. **No test data were used.**
> - Results involving SpatialNet-small and TF-GridNet come from **snapshots taken during training** (SpatialNet epoch 72, TF-GridNet epoch 16); they are **snapshot diagnostics only** and do not represent completed models.
> - All verdict thresholds are empirical; no significance tests were done (c6-real-v1 gives group-level bootstrap intervals, but judgements still follow the point estimates).
> - **The advantage of the audio LLM over "ASR + text LLM" has not been demonstrated.**

## 1. Confirmed results (obtained under rules written in advance; limited to the samples above)
| Item | Result | Document |
|---|---|---|
| Prompts | P2 (audio) and T2 (text) frozen. On clean speech, target-any of P2 is 0.640 (P0: 0.404) | `c1_p2t2_results.md` |
| C3 (single-channel mixture + direction instruction) | 0.125; c3-gate-v1 verdict **stop, insufficient evidence**. Moving on to the front end was the project lead's decision | `c3_results_v1.md` |
| Factor diagnosis | the main source is two-talker overlap (only 40 samples) | `c3_factor_diag_v1.md` |
| Front-end check | frontend-sanity-v1 found no pipeline bug; stream 0 of FaSNet-TAC has inverted polarity, and after correction direction-matching accuracy is 1.00 | `frontend_sanity_v1.md`, `frontend_mix_v1_eval_v3.md` |
| C4 / C5 (FaSNet-TAC) | 0.395 / 0.395; C4-direct 0.670. **For FaSNet-TAC, separation degradation is the bottleneck** | `c4_v1.md`, `c5_v1.md`, `c4_gap_diag_v1.md` |
| Stronger front ends (snapshot diagnostics) | SpatialNet ep72 + P2 0.635, Whisper + T2 0.590; TF-GridNet ep16 + P2 0.595, Whisper + T2 0.545. Both reach the empirical threshold for "a stronger front end clearly helps" | `frontend_compare_v1.md` |
| C7-W | Whisper + T2: C1 0.650, C4-direct 0.640, FaSNet C5 0.340; separated speech is also harder to transcribe (WER 0.45 vs 0.08) | `c7_whisper_v1.md` |
| C6δ (simulated offset, sensitivity analysis only) | with the query direction offset by ±20°, the stream-selection error rate peaks at about 0.40; the two front ends have almost identical stream-selection error rates; when SpatialNet selects the wrong stream, interferer-only is higher (0.245 at +20°) | `c6_delta_v1.md`, `c6_delta_spatialnet_v1.md` |
| **Real C6** (c6-real-v1) | SpatialNet + P2, natural angle weighting: interferer-only 0.049 for 8 sectors, 0.032 for σ = 10°; margin abstention (τ = 17°) and sector abstention B both bring it below 0.007, with weighted coverage around 0.85. Under the frozen rules the verdict is **"matters only at small angles or under some conditions"** | `c6_real_v1.md` |

## 2. Unconfirmed hypotheses
1. **The advantage of the audio LLM has not been demonstrated**: on the same front-end output, P2 is 0.030–0.055 above Whisper + T2 (SpatialNet snapshot C5 0.635 vs 0.590; C4-direct 0.670 vs 0.640; FaSNet C5 0.395 vs 0.340), with no significance tests and only on devel.
2. There is no causal evidence for the sources of the FaSNet-TAC residual (residual overlap, non-linear distortion or other).
3. Whether SpatialNet / TF-GridNet with completed training keep the level of the snapshots is unknown.
4. The severity of direction uncertainty is sensitive to sector granularity: interferer-only is 0.049 for 8 sectors (only 0.0011 below 0.05) and 0.075 for 4 sectors. How real users would give directions has not been measured.
5. The reason for the large gap of BLSTM+MVDR between mix_v1 and cv_dev93 has not been found; "domain difference" is a judgement reached by elimination.
6. The devel results may be optimistic: polarity correction, τ = 17° and so on were all fixed after looking at results on devel; devel has only 11 speakers.

## 3. New main line (decision of the project lead 2026-11-06)
1. **Front end**: SpatialNet-small is the main front end. After training completes, rerun with the completed checkpoint under the existing rules (cv_dev93 check → mix_v1 stream selection → P2 and Whisper + T2 on both streams → C5, C6δ, c6-real-v1); **on completion, report first and submit jobs only after approval**.
2. **Direction uncertainty**: kept as "a problem at small angles or with coarse input", not the current main focus.
   - margin abstention (τ = 17°) and sector abstention B are the **baselines** for all subsequent conditions;
   - the main conditions are 8 sectors and σ = 10°; **4 sectors is a sensitivity analysis only**;
   - C6δ remains a sensitivity analysis only.
3. **Downstream**: the main bottleneck shifts to downstream target entity extraction. A new LoRA plan draft is added (`docs/lora_slu_plan_draft.md`, plan only, not approved, not run), comparing P2, LoRA-fine-tuned Qwen and Whisper + T2.
4. **Control**: Whisper + T2 remains the "ASR + text LLM" control throughout, used to test whether the audio LLM has an advantage.

## 4. Changes to the original plan
| Original plan | Change | Reason |
|---|---|---|
| C6 = separated streams + stream selection by estimated azimuth | C6 = the user gives the target direction (a sector or an angle with error) + the system selects the stream by estimated source direction (`c6_real_v1.md`; `conditions.md` updated) | the mixture audio itself cannot tell whom the user wants to hear |
| M3 | **paused** (definition below this section) | real C6 judged to matter only at small angles or under some conditions |
| C7 = separated speech → ASR → text LLM | ASR fixed as Whisper large-v3 (revision `06f233fe…`), text LLM with the frozen T2; plan A with Qwen's own transcription is no longer the main control | plan A transcripts often turned into responses to the command (WER 3.96) |
| FaSNet-TAC as main front end | FaSNet-TAC becomes a control; **Whisper + T2 on the other stream is not run retroactively** | the snapshot front ends are clearly stronger; FaSNet is compared with P2 only |
| no fine-tuning | a LoRA plan draft is added, **still needing approval** before running | downstream entity extraction has become the main gap |
| no abstention | margin abstention and sector abstention B as baselines | both reduce interferer-only to less than half on the main conditions, with coverage ≥ 0.60 |

**Definition of M3** (supplementing the "M3 paused" row of the table above; source: the section "proposed method and ablation experiments" of the original plan PDF of 2026-11-03):

M3 (LLM-level fusion, candidate main method): pass both separated streams, the match of each stream to the target direction, and reliability information together to the LLM. The trainable parts are a reliability estimator + LoRA. Reliability features (first round): the difference between the target direction and each stream's estimated direction, the sharpness of the direction-spectrum peak, the similarity and residual crosstalk between the two outputs, and energy and intelligibility proxies of the separated streams. Input methods from low to high cost: both streams, directions and discrete reliability levels provided in the prompt → learnable reliability tokens → frame-wise weighting. Training target: structured output (intent, entities, confidence level or "cannot determine"), using flipped samples, supervised with "cannot determine" when the direction is indistinguishable. Controls: M1 hard stream selection (no training), M2 reliability-based stream selection (small reliability estimator).

The reason for pausing is unchanged: c6-real-v1 was judged to "matter only at small angles or under some conditions", and SpatialNet's stream-selection error rate on distinguishable samples is 0.

## 5. Unchanged constraints
- mix_v1, the P2/T2 prompts, slu-score-v2, binding-score-v1, the v3 stream-selection rule, c3-gate-v1 and all existing results and verdicts are not changed.
- test is not touched until the project lead decides otherwise.
- Training jobs are not touched; downloads, fine-tuning and GPU jobs all require reporting paths, disk and configuration in advance and waiting for approval.
