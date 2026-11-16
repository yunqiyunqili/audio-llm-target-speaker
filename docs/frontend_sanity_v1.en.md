English translation of frontend_sanity_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：frontend_sanity_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Front-end pipeline check v1 (frontend-sanity-v1; 2026-11-05; method and interpretation rules written before the run and not changed after the results were seen)

Purpose: determine whether the low results of frontend_mix_v1 (job 22014183) are due to a bug in the existing pipeline, or to short utterances or the SLURP domain difference. Qwen is not run, C4/C5 are not run; mix_v1, the existing front-end results and the frontend_mix_v1 verdict are not changed; `data/mix_v1` is not written.

Common settings: front ends BLSTM+MVDR and FaSNet-TAC, checkpoints read-only, sha256 and configuration checked before loading. The inference code is **line-by-line identical** to `scripts/sms_wsj/frontend_mix_v1.py`: read float32 (T, 6) with soundfile → convert to (1, T, 6) → `T.normalize` → `model(x) × scale`. SI-SDR uses `sisdr` and `scores()` from the original `tfgridnet/objectives.py` (PIT, two-talker mean). The 50 cv_dev93 groups are drawn with `random.Random(2026).sample(sorted(cv_dev93 ids), 50)`, and B and C share these 50 groups. Output is written to `outputs/frontend_sanity_v1/<jobid>/`, metrics and JSON only, **no audio written**.

## A. Inference pipeline check (SMS-WSJ test_eval92, first 50 utterances)
- Data: the dataset used by the original evaluation (`sms_wsj_direct_v1`, the first 50 in the order of `test_eval92.jsonl`); the reference is the mic 0 direct sound used by the original evaluation.
- First check: the mixture signal read directly with soundfile is exactly identical to the mixture signal returned by the original dataset's `__getitem__`.
- Compute si_sdr_db and si_sdri_db per utterance with the inference code above and the original `scores()`, and compare item by item with the original `test_utterances.jsonl`.
- **Rule**: if either front end meets any of the following — maximum per-item difference > 0.05 dB, mean difference > 0.01 dB, any PIT permutation mismatch — the verdict is **the existing inference pipeline has a problem**.

## B. Mixing-script check (cv_dev93, 50 groups)
- **Our pipeline**: using the original WSJ utterances (`original_source`, already 8 kHz, so the 16k→8k step is skipped) and the corresponding h_0, h_1, apply the same flow as `mix_slurp_v1`: rir_start, 50 ms early, alignment with `extract_piece(offset − rir_start)`, the official offset and the official length T; each source scaled to active-frame RMS = −30 dBFS of the mic 0 early image (SIR 0 dB); white noise added at the official SNR, with the noise seed from `np.random.RandomState(2026 + 组序号)`. The reference is the early image (mic 0) computed by our pipeline. Generated only in memory.
- **Official control**: the official observation and the official early image (`speech_reverberation_early`, mic 0); results with the official direct sound as the reference are also reported (`sms_wsj_direct_v1/cv_dev93`).
- Metric: two-talker mean SI-SDRi (PIT, early reference). The distribution of the official SIR (log_weights) is also reported.
- **Rule**: if either front end has |mean SI-SDRi(our pipeline) − mean SI-SDRi(official)| > 2.0 dB, the verdict is **the mixing pipeline or the level rule has a problem**.

## C. Duration check (cv_dev93, the same 50 groups, official mixtures and official references)
- Truncation length: for each group, draw one L with `np.random.default_rng(2026)` from the 200 T_8k of mix_v1 (if L ≥ the original length, do not truncate, and record this).
- Truncation window: slide in steps of 0.1 s (800 samples) and choose the position with the most two-talker active-frame overlap (active frames computed on mic 0 of the official early image, 256-point frames, peak −40 dB); the observation and reference signals are cut at the same segment.
- Comparison: the two-talker mean SI-SDRi (early reference) before and after truncation, and the mean of mix_v1 (job 22014183: BLSTM+MVDR 2.90, FaSNet-TAC 0.47).
- Let gap = SI-SDRi(official full) − SI-SDRi(mix_v1), drop = SI-SDRi(official full) − SI-SDRi(official truncated).
- **Rule** (judged per front end): if drop ≥ 0.5 × gap and gap > 2 dB, the verdict is **short utterances are the main cause**; otherwise **short utterances are not the main cause**. If gap ≤ 2 dB, report only, no judgement.

## D. θ1/θ2 asymmetry check (reads only the existing outputs of 22014183 and mix_v1)
- Estimate directions with `doa()` in `eval_frontend_mix_v1.py` for: mic 0 of the observed mixture, the oracle early images (A and B), and the two output streams of each front end. For each object, count whether the estimated direction is closer to θ1 or θ2 (compared by |wrap(θ̂ − θ)|, ties recorded as ties).
- Report the proportion "closer to θ1" grouped by s1/s2, by talker A/B (streams assigned by reference matching), and by angle stratum. 95% confidence intervals: group-unit bootstrap, 10,000 times, `numpy default_rng(2026)`.
- Structural checks:
  - the observation file is 8 kHz, 6 channels, shape (T, 6), with T matching the manifest;
  - the output files are 8 kHz, single-channel, length T;
  - channel order: correlate each channel with the corresponding channel of early_A + early_B; the maximum correlation must fall on the channel with the same index;
  - the energy ratio of the two FaSNet-TAC streams and the number of nearly silent samples (RMS of the weaker stream < 1e-4 × the stronger).
- Explain the difference between the 77.9° target-stream error and 1.9° interferer-stream error of FaSNet-TAC, and why [30,60) is below chance: look at where the directions of the two streams fall (θ1, θ2 or elsewhere), compared with the direction of the observed mixture.
- **Rule**: if any of the following holds, the verdict is **the existing pipeline has a bug**:
  - any structural check fails;
  - the proportion of oracle early images whose estimated direction is closer to their own position (A: θ1 in s1, θ2 in s2; B the reverse) < 0.95.

## Conclusion (choose exactly one)
Check in the order A → B → D:
1. A or D judged a bug: **the existing inference pipeline has a bug**; list the location, do not fix it on one's own initiative, and mark the mix_v1 front-end results as void.
2. B judged to have a problem: **the mixing script or level has a bug**; handled as above.
3. A, B and D all normal: **the pipeline is normal; it is mainly short utterances or the SLURP domain difference**. The result of C states whether short utterances are the main cause; if not, it is attributed to the SLURP domain difference (including factors other than short utterances, such as speaking style, recording conditions and level distribution).
4. Any item could not be completed: **cannot be judged for now**; state what is missing.

All thresholds are empirical, not significance tests.

## Implementation
- `scripts/sms_wsj/frontend_sanity_v1.py` (A, B, C; GPU; tfgridnet environment; front end chosen with `--model`) and `.sbatch`. Job configuration: <SLURM_ACCOUNT>, <GPU_PARTITION>, 1 A100, 4 CPU, 32 GB, 1 hour.
- `scripts/sms_wsj/frontend_sanity_d.py` (D; CPU; read-only).

## Results (2026-11-05)
- Run records:
  - A/B/C: GPU job 22014492, <NODE>, 13:05–13:07, exit code 0. The commit recorded by the job is `d107701`, DIRTY = 1; the uncommitted file was the D script, not yet committed at the time, on which A/B/C do not depend. CPU dry-run srun 22014488.
  - D and the final summary: srun 22014499 (<CPU_PARTITION>); the commit of the D script is `18b4150`.
- Results in `outputs/frontend_sanity_v1/22014492/`: `sanity_{blstm_mvdr,fasnet_tac}.json`, `sanity_D_and_decision.json`, `sanity_D.log`. Metrics only; no audio written.

**A. Inference pipeline** (first 50 of test_eval92): for both front ends, the maximum per-item difference from the original records is 0.0 dB, the mean difference 0.0 dB, with no PIT permutation mismatch; the mixture signal read directly with soundfile is exactly identical to the original dataset loading. → **no problem**.

**B. Mixing script** (50 cv_dev93 groups, early reference, two-talker mean SI-SDRi)
| | Remixed with the mix_v1 pipeline | Official mixture | Difference | Official mixture, direct-sound reference |
|---|---|---|---|---|
| BLSTM+MVDR | 6.62 | 6.94 | −0.33 | 4.17 |
| FaSNet-TAC | 1.43 | 1.39 | +0.04 | 15.05 |
- SI-SDR of the mixture signal: mix_v1 pipeline −0.94, official −0.92. The median difference between the two talkers' official log_weights is 1.65 dB, maximum 4.59 dB.
- Both front ends have differences ≤ 2 dB → **no problem**.

**C. Duration** (the same 50 groups, truncated to 1.7–4.3 s, taking the window with the most two-talker overlap)
| | Full → truncated (early) | Full → truncated (direct sound) | gap (official full − mix_v1) | drop | Verdict |
|---|---|---|---|---|---|
| BLSTM+MVDR | 6.94 → 5.45 | 4.17 → 3.45 | 4.05 | 1.50 (37% of gap) | short utterances are not the main cause |
| FaSNet-TAC | 1.39 → 1.07 | 15.05 → 14.16 | 0.91 | 0.31 | gap ≤ 2 dB, report only |

**D. θ1/θ2 asymmetry**
- Structural checks: observation files 200/200 correct, channel order 200/200 correct, output shapes of both front ends 200/200 correct. → **pass**.
- Oracle early images judging their own position: A 1.00, B 0.985 (95% confidence interval [0.965, 1.0]). → **pass**.
- Proportion with estimated direction closer to θ1: observed mixture 0.48 [0.43, 0.54]; BLSTM+MVDR stream A and stream B both 0.47; FaSNet-TAC stream A 0.41 [0.34, 0.47], stream B 0.40 [0.33, 0.47].
- Looking at the swapped samples separately, both streams follow the position: in s1, the proportion of stream A closer to θ1 is 0.62 for BLSTM+MVDR and 0.58 for FaSNet-TAC; in s2, the proportion of stream B closer to θ1 is 0.55 for BLSTM+MVDR and 0.59 for FaSNet-TAC.
- The two FaSNet-TAC streams:
  - the stream assigned as the interferer stream falls near θ2 (±10°) in 100/200, so the interferer-stream error is small;
  - the stream assigned as the target stream falls near neither θ1 nor θ2 in 116/200, so the target-stream error is large (median 77.9°);
  - the median RMS ratio of the two streams is 0.80, with 4 nearly silent samples.
- Why [30,60) is below chance: in this stratum the interferer stream falls at θ2 in 35/50, while the target stream has no clear direction in 35/50; θ2 is only 30–60° from θ1, so direction matching tends to pick the interferer stream.
- BLSTM+MVDR: in 147 of the 200 samples the directions of both streams fall on the same side (both near θ1 or both near θ2).

## Conclusion (exactly one chosen by the rules)
**The pipeline is normal; it is mainly short utterances or the SLURP domain difference.**
- For BLSTM+MVDR: short utterances explain only 37% of the gap, so by the rules it is attributed to the **SLURP domain difference**.
- **This is a judgement reached by excluding other causes, not a direct proof**: A, B and D found no problem, and C is not enough to explain the gap, so it is attributed to the domain difference; but the "domain difference" itself was not measured separately.
- Another finding outside the frozen options: FaSNet-TAC was trained on the direct sound; with the early reference it reaches only 1.39 dB even on the official mixtures, versus 15.05 dB with the direct-sound reference. So the early and full-reference results of FaSNet-TAC in frontend_mix_v1, and the reference matching based on those two references, do not represent its separation ability. Subsequent evaluation switches to the direct-sound reference; see `docs/frontend_mix_v1_eval_v2.md`.
- Kept unchanged: the v1 results and verdict of frontend_mix_v1, and the mix_v1 audio.
