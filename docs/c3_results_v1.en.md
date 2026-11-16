English translation of c3_results_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c3_results_v1.md

# C1 / C3 results (job 22014089; P2 prompt; c3-gate-v1; 2026-11-05)

## Scope
- Only the 100 devel groups × 2 swapped samples of mix_v1, run once. Not representative of overall performance, and not test results.
- The input of C3 is the **mic 0 single-channel** mixture signal; the audio itself contains no multichannel information that could be used to judge azimuth.
- Rules: `docs/c3_gate_v1.md` (frozen before job submission); scoring rules slu-score-v2 and binding-score-v1, both unchanged.

## Run record
- Job 22014089, node <NODE>, 2026-11-05 11:56:02–12:04:50 (8 min 48 s), exit code 0, peak process memory (MaxRSS) 26.5 GB.
- Loading 54.6 s. Inference C1 143.3 s, C3 233.2 s; inference GPU memory peak about 16.9 GiB.
- Code version: commit `233ba7b`, no uncommitted dependency files at run time. The sha256 of `qwen_omni_c3_p2.py` is `568edf98…`; the full prompt is recorded in `outputs/qwen_omni_c3_p2/22014089/config.json`.
- CPU dry-run: srun 22014088. Evaluation: `scripts/qwen/eval_c3_v1.py`, results in `outputs/qwen_omni_c3_p2_eval/22014089/`, reading only the existing raw outputs, run on the login node.

## C1 (124 clean recordings, P2 prompt)
- Strict intent accuracy 0.766; value-only P / R / F1 0.694 / 0.597 / 0.642; strict full-semantics accuracy 0.25.
- 1 zero-entity output, 0 parse failures, 0 refusals; 2 out-of-inventory intents, 7 out-of-inventory entity types.
- Each recording is computed against its opponents in each group, then averaged over recordings: target-any 0.694, all target values hit 0.571.
- Compared with the P2 results of job 22011390: for the 50 overlapping recordings, **raw outputs 50/50 exactly identical, parsed entities 50/50 identical**.

## C3 (200 mixture samples, frozen direction instruction + P2 prompt)
| Item | Result |
|---|---|
| Four class proportions | target only 0.105, interferer only 0.115, both 0, neither 0.780 |
| Target hits | target-any 0.125, all target values hit 0.105 |
| Single-talker verdicts / target share | 44 / 0.477, 95% confidence interval [0.391, 0.564] (group-unit bootstrap, 10,000 times, seed 2026) |
| Pair-flip accuracy | 0 / 100 groups |
| Whether the person the answer points to changes between swaps | changed in 13 groups; of the 87 unchanged groups, neither both times 71, A only both times 10, B only both times 6 |
| Parsed entity values exactly identical between swaps | 25 groups |
| Parse failures / refusals | 0 / 0 |
| Out-of-inventory intents / entity types | 16 / 29 |
| SLU metrics against the target | intent 0.355, value-only F1 0.117, WER 0.89 |

**Gate verdict (c3-gate-v1): stop. Only 44 single-talker verdicts, fewer than 50; too few samples to judge.** Front-end inference was not entered.

## Strata (exploration only; each stratum has only 3–29 single-talker verdicts)
| Stratum | Samples | Target only / interferer only / neither | Single-talker verdicts | Target share |
|---|---|---|---|---|
| angle [0,15) | 50 | 0.14 / 0.18 / 0.68 | 16 | 0.438 |
| angle [15,30) | 50 | 0.06 / 0.08 / 0.86 | 7 | 0.429 |
| angle [30,60) | 50 | 0.14 / 0.10 / 0.76 | 12 | 0.583 |
| angle [60,180] | 50 | 0.08 / 0.10 / 0.82 | 9 | 0.444 |
| T60 [0.2,0.3) | 68 | 0.147 / 0.118 / 0.735 | 18 | 0.556 |
| T60 [0.3,0.4) | 44 | 0.114 / 0.182 / 0.705 | 13 | 0.385 |
| T60 [0.4,0.5] | 88 | 0.068 / 0.080 / 0.852 | 13 | 0.462 |
| overlap [0,0.5) | 9 | 0.222 / 0.111 / 0.667 | 3 | 0.667 |
| overlap [0.5,0.75) | 72 | 0.139 / 0.153 / 0.708 | 21 | 0.476 |
| overlap [0.75,0.9) | 91 | 0.077 / 0.099 / 0.824 | 16 | 0.438 |
| overlap [0.9,1] | 28 | 0.071 / 0.071 / 0.857 | 4 | 0.500 |
| full-image SIR < −0.5 dB | 25 | 0.08 / 0.24 / 0.68 | 8 | 0.250 |
| full-image SIR [−0.5, 0.5] dB | 148 | 0.088 / 0.108 / 0.804 | 29 | 0.448 |
| full-image SIR > 0.5 dB | 27 | 0.222 / 0.037 / 0.741 | 7 | 0.857 |

## Observations (description only, no conclusions)
- Some C3 outputs match neither sentence: e.g. g000_s1 output a Chinese transcript; both swapped samples of g001 were output as "set … alarm".
- Further breakdown and diagnosis below.

## Output breakdown and transcript choice (reads only the outputs of 22014089; `scripts/qwen/c3_output_diag.py`, results in `outputs/qwen_omni_c3_p2_eval/22014089_output_diag/`)
- 156 "neither" items in total. The class rules were written in the script first and are applied in the order "non-English → only partial entity values → transcript close → unrelated to both sentences", each item going into one class only:
  - unrelated to both sentences 116;
  - transcript close to the target or the interferer (WER < 0.5) but no entities extracted 24;
  - non-English 12;
  - only partial entity values 4.
- Clear choice: WER < 0.5 against the closer talker, and the two WERs differ by ≥ 0.1. 52 of the 200, of which target 26 and interferer 26. By full-image SIR stratum:

| SIR stratum | Samples | Clear choice | Chose target | Chose interferer |
|---|---|---|---|---|
| < −0.5 dB | 25 | 6 | 1 | 5 |
| −0.5 to 0.5 dB | 148 | 40 | 19 | 21 |
| > 0.5 dB | 27 | 6 | 6 | 0 |

- The above is diagnosis only, not a gate conclusion.

## Diagnostic job 22014131 (20 groups, 40 samples, seed 2026; three prompts on the same audio)
- Script: `scripts/qwen/qwen_omni_c3_diag.py`; evaluation: `scripts/qwen/eval_c3_diag.py`, results in `outputs/qwen_omni_c3_diag_eval/22014131/`.
- Run: commit `bd389ae`, no uncommitted dependency files at run time; node <NODE>, 12:13:39–12:19:23, exit code 0; the CPU dry-run used srun 22014130.
- Groups drawn: 13, 15, 28, 40, 53, 56, 62, 64, 65, 70, 71, 73, 75, 76, 79, 82, 93, 96, 98, 99.

| | D1 only asks "Transcribe all speech in this audio." | D2 P2, without the direction instruction | D3 frozen C3 prompt |
|---|---|---|---|
| Agreement with 22014089 | — | — | raw outputs 40/40 exactly identical |
| Refusal or chit-chat | 17 say "can't transcribe", 36 exceed 25 words | 0 | 0 |
| WER < 0.5 against some talker | 0 | 11 | 11 |
| Clear choice (target / interferer) | 0 / 0 | 5 / 6 | 5 / 6 |
| Unrelated to both sentences | — | 26 | 26 |
| Target only / interferer only / neither | — | 4 / 5 / 31 | 4 / 4 / 32 |

- D2 vs D3: the person the answer points to is the same in 39/40, the clear choice the same in 38/40, but only 14/40 raw outputs are exactly identical.
- For the two swapped samples of D2, only 1/20 groups have exactly identical raw outputs.
- Interpretation (inference, not an experimental conclusion): removing the direction instruction leaves the degree of anomaly almost unchanged, so it does not look as if the direction instruction causes it. D1 mostly refuses or chats on the mixture audio and cannot serve as a pure-transcription control. The mixture audio itself is most likely the main cause, but the factors of overlap, reverberation and 8 kHz bandwidth have not yet been separated; see `docs/c3_factor_diag_v1.md`.

## Remaining capacity beyond mix_v1 (`scripts/slurp/mix_v1_remaining_capacity.py`, srun 22014133; statistics only, no audio generated)
- Of the 248 pairs satisfying all filters, mix_v1 uses 100, leaving 148; of these, 36 pairs have both recordings unused by mix_v1.

| Basis | Candidate pairs | Maximum groups drawable (exact maximum) | Maximum male groups when groups are at maximum |
|---|---|---|---|
| a separate new set, each recording at most 3 times, each person at most 30 groups | 148 | 74 | 2 |
| only recordings entirely unused by mix_v1 | 36 | 31 | 0 |
| sharing the above quotas with mix_v1 | 148 | 0 | — |
