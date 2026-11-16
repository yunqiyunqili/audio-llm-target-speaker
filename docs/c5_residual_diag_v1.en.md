English translation of c5_residual_diag_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c5_residual_diag_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C5 residual decomposition v1 (c5-residual-diag-v1; 2026-11-05; read-only, written before computation; descriptive only, **not written as causal**)

- Object: the 200 target streams of C5 FaSNet-TAC. Used after the C5 polarity correction, 8 kHz, the same as separation-quality-diag-v1.
- References (mic 0, with the gains and alignment of mix_v1):
  - direct: `outputs/frontend_mix_v1_eval_v2/direct_refs/`;
  - early: the `early_{A,B}_6ch_8k.wav` already written by mix_v1;
  - full image: recomputed in memory, by the same method as eval v2.
- Each source is split into three parts: direct sound d, early reflections r_e = early − d, late reverberation r_l = full − early.
- Nested least squares (all signals zero-mean first):
  1. basis B1 = {d_T, d_I}, residual energy R1;
  2. basis B2 = B1 ∪ {r_e,T, r_e,I}, residual energy R2;
  3. basis B3 = B2 ∪ {r_l,T, r_l,I}, residual energy R3.
- Descriptive quantities:
  - proportion explainable by early reflections = (R1 − R2) / R1;
  - proportion explainable by late reverberation = (R2 − R3) / R1;
  - proportion still unexplained = R3 / R1.
  - Also reported, for reference: the additional proportion explained, relative to direct, by early reflections only or by late reverberation only, computed with bases {d, r_e} and {d, r_l} respectively.
- Notes:
  - early reflections, late reverberation and direct sound are strongly correlated, and the least-squares allocation is not unique; **these proportions only describe "how much of the residual can be represented linearly by these components"**;
  - no judgement is made about what causes the residual.
- Report the median and quartiles over all 200 samples, and separately for the two groups split by C5 target-any.
- Listening samples: draw 10 with `random.Random(2026).sample(sorted(200 个 key), 10)` and list:
  - the file path of the C5 target stream (`data/mix_v1/separated_v1/fasnet_tac/gNNN_sS/out{k}.wav`, original polarity);
  - the target's direct reference. It is currently in npz format; for listening, these 10 are exported as wav (8 kHz, float32) into `outputs/c5_residual_diag_v1/listen/`, without writing to `data/`.
- Implementation: `scripts/sms_wsj/c5_residual_diag.py` (tfgridnet environment, CPU, srun).

## Results (2026-11-05; a descriptive least-squares decomposition with strongly correlated components, **not causal**)
- Run record: srun 22015449 (<CPU_PARTITION>, tfgridnet environment), script commit `4caeb4b`. Results in `outputs/c5_residual_diag_v1/{residual.json,per_sample.jsonl}`; the listening wav files are in `listen/`.

| Proportion of the "residual energy after direct projection" | All 200 (P25 / median / P75) | C5 hit (79) median | C5 miss (121) median |
|---|---|---|---|
| linearly explainable by early reflections | 0.018 / **0.047** / 0.086 | 0.045 | 0.049 |
| further linearly explainable by late reverberation | 0.0014 / **0.0043** / 0.0095 | 0.0043 | 0.0036 |
| still unexplained | 0.905 / **0.945** / 0.972 | 0.948 | 0.941 |

- With late reverberation only ({d, r_l}), the median additional proportion explained is 0.0040. With early reflections only, it is the same as the nested decomposition, median 0.047.
- Description: in the residual of the C5 target stream, about 95% (median) cannot be represented linearly by the direct sound, early reflections or late reverberation of the target or the interferer. These proportions differ little between the hit and miss groups.
- Listening samples (`random.Random(2026).sample`, 10 in total; the C5 target-stream files are at `data/mix_v1/separated_v1/fasnet_tac/<key>/out{k}.wav`, original polarity; the direct references are at `outputs/c5_residual_diag_v1/listen/<key>_direct_target_8k.wav`):

| key | C5 target stream | C5 hit? | Target sentence |
|---|---|---|---|
| g015_s1 | out1 | no | where is george w bush from |
| g040_s2 | out0 | no | i like to listen to jazz |
| g064_s1 | out0 | no | how tall is micheal jackson |
| g065_s2 | out0 | yes | what do i have scheduled for next week |
| g082_s2 | out1 | yes | what is a basketball |
| g013_s1 | out1 | no | show results for cooking lasagna from web |
| g028_s2 | out0 | yes | can you tell me what haggis looks like |
| g076_s2 | out1 | no | start playing jazz music |
| g079_s2 | out0 | no | what are my reminders today |
| g071_s1 | out1 | no | today's weather is awesome |
