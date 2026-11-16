English translation of c3_factor_diag_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c3_factor_diag_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Factor diagnosis of the sources of C3 anomalies v1 (c3-factor-diag-v1; 2026-11-05; conditions and interpretation rules written before job submission and not changed after the results were seen)

Purpose: distinguish which factor the anomalous C3 outputs more likely come from — bandwidth (8 kHz), early reverberation, full reverberation, two-talker overlap, or the P2 constraints. **Diagnosis only**; not a gate, and no c3_gate_v2 is written. The separation front end is not run, and mix_v1, c3_gate_v1, the P2 prompt, binding-score-v1 and existing results are not changed.

## Conditions
All conditions use the frozen P2 prompt (`scripts/qwen/qwen_omni_c1_p2t2.py: PROMPT_P2`), except X3, which uses P0 (`scripts/qwen/qwen_omni_c1_semantic.py: PROMPT`). Decoding settings as before: greedy decoding, max_new_tokens 256, batch 1, text output only. Audio is **generated only in memory**, with no files written.

**Single-talker conditions: the 124 recordings in mix_v1**
| Condition | Audio |
|---|---|
| C1 (reference, existing) | original 16 kHz (C1 of job 22014089) |
| C1b | `qwen_omni_single.c1b`: resample_poly first 1/2 then 2/1, truncated back to the original length; level unchanged |
| C2 | first down to 8 kHz (resample_poly 1/2), then convolved with the early RIR of mic 0 (rir_start the minimum over the 6 channels, early = the 400 samples after rir_start, i.e. 50 ms), aligned by 0 − rir_start (extract_piece, length = length of the 8 kHz signal), then scaled to active-frame RMS = −30 dBFS of the mic 0 early image, and finally resample_poly 2/1 up to 16 kHz, truncated to 2T. The active-frame definition is the same as mix_v1. This flow follows the definition of C2 in `docs/conditions.md` |
| R1 | the same as C2, but with the **full** reverberation of the same RIR; the gain follows that computed from the early image (same as mix_v1); no interferer, no noise |

- RIR: each recording uses the cv_dev93 RIR of the group in which it **first appears** in mix_v1 (the smallest group number). The position is taken from swap 1 of that group: A uses h_0, B uses h_1.

**Mixture conditions: the same 40 samples used by 22014131 (20 groups × 2 swapped samples)**
| Condition | Audio | Prompt |
|---|---|---|
| D2 (reference, existing) | the existing mic0_16k mixture audio (reverberation + noise + 8 kHz bandwidth) | P2, without the direction instruction |
| X1 | the two talkers' **dry** speech, 16 kHz, mixed with the mix_v1 offsets (8 kHz offset × 2); each scaled to active-frame RMS = −30 dBFS (hence 0 dB); no noise | P2 |
| X2 | X1 first down to 8 kHz and back up to 16 kHz (resample_poly 1/2 then 2/1, truncated back to the original length) | P2 |
| X3 | the existing mic0_16k mixture audio | **P0** |

- Dry speech has no position, so in X1 and X2 the audio of s1 and s2 of the same group is exactly the same; only the "target" label differs (A for s1, B for s2).

## Metrics (reported for every condition)
- target-any, target only, neither (binding-score-v1, parsed entity values only).
  - Single-talker conditions: for each recording, computed against its opponents in each group, then averaged over recordings, the same as for C1 in `eval_c3_v1.py`.
  - Mixture conditions: per sample, the target of s1 is A and of s2 is B.
- **M = number of items with WER < 0.5 against either talker** (computed after normalize_v2; in single-talker conditions there is only the recording's own speaker).
- **U = number of items unrelated to both sentences**: no parsed entity value equals a reference value, and the WER against every relevant talker is ≥ 0.5.
- Number of non-English items (non-ASCII letters in the transcript or raw output); number of refusals (no_json and matching `postproc_c1_ext.REFUSAL`); number of parse failures.
- X1, X2 and X3 are each compared with D2.

## Interpretation rules (written in advance; using the proportion m = M / number of samples, threshold Δ = 0.15, absolute)
| Factor | Comparison | Condition for "the factor has an effect" |
|---|---|---|
| bandwidth (single-talker) | C1 → C1b | m(C1) − m(C1b) ≥ 0.15 |
| early reverberation (single-talker) | C1b → C2 | m(C1b) − m(C2) ≥ 0.15 |
| late reverberation (single-talker) | C2 → R1 | m(C2) − m(R1) ≥ 0.15 |
| two-talker overlap (dry speech) | C1 (the C1 results of the target recordings of these 40 samples) → X1 | m(C1_40) − m(X1) ≥ 0.15 |
| bandwidth (overlapped speech) | X1 → X2 | m(X1) − m(X2) ≥ 0.15 |
| reverberation + noise (overlapped speech) | X2 → D2 | m(X2) − m(D2) ≥ 0.15 |
| P2 constraints | D2 ↔ X3 | \|m(X3) − m(D2)\| ≥ 0.15, or \|u(X3) − u(D2)\| ≥ 0.15 (u = U / 40); the direction of change is also reported |

- **Main source**: among the factors meeting the condition, the one with the largest drop in m (the P2 constraints enter the comparison with the larger of \|Δm\| and \|Δu\|). If no factor meets the condition, the conclusion is "cannot be distinguished".
- When a difference is below the threshold, one can only say "the factor cannot be judged to have an effect", **not** "the factor has no effect".
- The threshold is empirical, not a significance test. The mixture conditions have only 40 samples, so 0.15 is about 6 items.
- Known confounder: C1 and C1b keep the original level, whereas C2, R1, X1 and X2 are uniformly scaled to −30 dBFS, so level differences are not controlled separately.

## Implementation
- Job script: `scripts/qwen/qwen_omni_c3_factor.py` and `.sbatch`; evaluation script: `scripts/qwen/eval_c3_factor.py`.
- Output directory: `outputs/qwen_omni_c3_factor/<jobid>/`.

## Results (job 22014155; commit 9c813ea, no uncommitted dependency files at run time; <NODE>, 12:27–12:38, exit code 0; CPU dry-run srun 22014153)
Output: `outputs/qwen_omni_c3_factor/22014155/`; evaluation: `scripts/qwen/eval_c3_factor.py`, results in `outputs/qwen_omni_c3_factor_eval/22014155/`.

| Condition | Samples | target-any | Target only | Neither | M (WER < 0.5 against either talker) | U (unrelated to both sentences) | Non-English | Refusals |
|---|---|---|---|---|---|---|---|---|
| C1 (existing) | 124 | 0.694 | 0.571 | 0.425 | 117 | 6 | 0 | 0 |
| C1b | 124 | 0.677 | 0.571 | 0.425 | 113 | 10 | 0 | 0 |
| C2 | 124 | 0.613 | 0.499 | 0.497 | 113 | 10 | 0 | 0 |
| R1 | 124 | 0.565 | 0.466 | 0.534 | 111 | 12 | 1 | 0 |
| D2 (existing) | 40 | 0.125 | 0.100 | 0.775 | 11 | 26 | 0 | 0 |
| X1 | 40 | 0.325 | 0.300 | 0.375 | 18 | 12 | 0 | 0 |
| X2 | 40 | 0.300 | 0.275 | 0.425 | 18 | 10 | 0 | 0 |
| X3 (P0) | 40 | 0.050 | 0.025 | 0.925 | 12 | 28 | 1 | 0 |

No condition has parse failures. In X1 and X2 the audio of s1 and s2 of the same group is exactly the same, and the outputs are also 20/20 identical.

**Judged under the frozen rules (Δ = 0.15, comparing m = M / number of samples):**
| Factor | Comparison | Change in m | Threshold reached? |
|---|---|---|---|
| bandwidth (single-talker) | C1 → C1b | 0.944 → 0.911 (−0.032) | no |
| early reverberation (single-talker) | C1b → C2 | 0.911 → 0.911 (0) | no |
| late reverberation (single-talker) | C2 → R1 | 0.911 → 0.895 (−0.016) | no |
| **two-talker overlap** | C1_40 → X1 | 0.95 → 0.45 (−0.50) | **yes** |
| bandwidth (overlapped speech) | X1 → X2 | 0.45 → 0.45 (0) | no |
| **reverberation + noise (overlapped speech)** | X2 → D2 | 0.45 → 0.275 (−0.175) | **yes** |
| P2 constraints | D2 ↔ X3 | Δm = +0.025, Δu = +0.05 | no |

**Main source (by the rules): two-talker overlap.**

## Diagnostic item: drop in entity hits under single-talker reverberation (recorded separately; no rule changed)
- Under the single-talker conditions, the proportion with "WER < 0.5 against the talker" hardly changes (C1b 0.911 → C2 0.911 → R1 0.895), but target-any decreases: C1 0.694, C1b 0.677, C2 0.613, R1 0.565.
- The frozen rules judge only by m, so this drop does not enter the verdict. It is only recorded here.

## Decision (2026-11-05, project lead)
- **The result of c3_gate_v1 remains "stop, insufficient evidence"** (44 single-talker verdicts, fewer than 50); it **did not pass**.
- **The project lead decided to proceed to the separation front end**, based on two-talker overlap being the main source in this diagnosis. This is the lead's decision, not the result of passing the gate.
- **The leakage check is not yet complete**; reporting by full-image SIR stratum continues.
- The drop in entity hits under single-talker reverberation is recorded separately as a diagnostic item, per the previous section.
