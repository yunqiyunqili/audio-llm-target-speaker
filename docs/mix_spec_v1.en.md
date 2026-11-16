English translation of mix_spec_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：mix_spec_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# Mixture pairing specification v1 (mix_spec_v1; 2026-11-05)

Status: pair filtering done; the C3 direction instruction frozen (2026-11-05); the list redrawn under the second-version constraints below (v2, superseding the first version); mixing rules written, with a dry-run that writes no audio.

## Data sources
- Speech: only recordings in split_v2_seed2026 **devel** (basis A). Some of the sentences come from the official train and were put into devel by the v2 rules (rules in `docs/provenance.md`). **test is not used.**
- RIRs: only SMS-WSJ **cv_dev93** (development set; 8 kHz, 6 microphones, each sample with two source positions h_0, h_1). **test_eval92 is not used.**

## Pair filtering rules
Base conditions: v2 devel candidate pairs, same gender, duration difference ≤ 1 s, each recording used at most 3 times; D_T and D_I both non-empty (definition in `docs/binding_score_v1.md`, value multisets after normalize_v2).

**Additional exclusion**: a pair is excluded if there exists any v ∈ D_T, u ∈ D_I satisfying either of the following:
1. **Same stem**: the word sequences after stemming each word are the same. Stemming rule (a simple rule, not the Porter stemmer):
   - first remove a word-final `'s`;
   - if length > 4 and ending in `ies`, change it to `y`;
   - otherwise, if length > 4 and ending in s/x/z/ch/sh plus `es`, remove `es`;
   - otherwise, if length > 3, ending in `s` and not ending in `ss`, remove `s`.
2. **Containment**: the complete word sequence of one side appears as a contiguous subsequence in the other (e.g. "john" and "tara john").

Implementation: `scripts/slurp/devel_capacity_mixspec_v1.py`.

## Capacity after exclusion (srun 22011159, <CPU_PARTITION>; statistics only, no sampling, no audio generated)
- 270 pairs under the base conditions, 22 excluded: 21 for the same stem (all today and today's, tomorrow and tomorrows), 1 for containment (john and tara john). 248 pairs remain, involving 188 recordings.
- With each recording used at most 3 times:
  - the exact maximum from 0/1 integer programming is **185 pairs** (HiGHS reports optimal), 12 fewer than the 197 before filtering;
  - greedy pairing gives 182 pairs, which is only a lower bound.
- In one optimal solution of the integer program:
  - 186 recordings and 11 speakers are used (6 female, 5 male);
  - by gender, 159 female pairs and 26 male pairs;
  - 27 intents covered; the largest reduction is datetime_query (19 → 8 pairs);
  - by sentence source, train+train 86 pairs, train+devel 79 pairs, devel+devel 20 pairs.
- Output: `outputs/mix_spec_v1_capacity/devel/capacity_mixspec_v1.json`, containing the original sentences of all 22 excluded pairs.
- Known limitation: this stemming rule treats "news" and "new" as the same stem. There is no such case among the 22 pairs excluded this time.

## Sampling list, first version (void)
- First version: greedy sampling with `random.Random(2026)`, result `configs/mix_v1_devel100.json` (srun 22012052). Reason for voiding: speakers and gender were too concentrated, with female voices in 90 groups and FE-250 appearing in 50 groups. This list was never used to generate audio.

## Sampling list, second version (constraints written before sampling, 2026-11-05)
**Constraints (all must be satisfied, not relaxed):**
1. Select only from the 248 pairs remaining after the filtering above (same gender, duration difference ≤ 1 s, D_T and D_I both non-empty, no shared stem or containment).
2. Each recording used at most 3 times.
3. **Each speaker appears in at most 30 groups** (counting both as A and as B).
4. **At least 20 male groups** (pairs are always same-gender, so a group is either a male group or a female group).
5. 100 groups in total. No repeated RIR samples; angles in four strata [0,15), [15,30), [30,60), [60,180], 25 groups each.

**Sampling method (seed 2026):**
- Sort the 248 pairs by (file_a, file_b) and shuffle with `random.Random(2026)` to obtain a random rank r (0 to 247) for each pair.
- Solve a 0/1 integer program: subject to constraints 2–4 and "exactly 100 groups", maximise Σ (248 − r)·x, i.e. among feasible solutions prefer pairs early in the random order (scipy milp / HiGHS).
- If this program is infeasible, do not relax the constraints; instead report the closest feasible results and stop:
  - (a) the maximum number of groups under constraints 2–4;
  - (b) with 100 groups under constraints 2–3, the maximum achievable number of male groups and the minimum achievable maximum speaker occurrence.
- After the 100 pairs are selected, the same rng then determines, in order: A/B direction, assignment of angle-stratum labels, RIR sample within each stratum, offset, noise SNR and noise seed. These rules are the same as in the first version.
- **Angle stratification is for diagnosis only**: 25 groups in each of the four strata, which does not represent the natural distribution of cv_dev93. Among the 982 cv_dev93 samples, [0,15) has 78, [15,30) 96, [30,60) 155, [60,180] 653.
- θ1 = the azimuth of position 0 (h_0) relative to mic 0, θ2 = the azimuth of position 1 (h_1). The azimuth definition is in `docs/conditions.md`: 0° is the direction from the array centre to mic 0, in (−180°, 180°].
- Implementation: `scripts/slurp/make_mix_list_v1.py` (v2 mode), output to `configs/mix_v1_devel100_v2.json`.

## Swapped samples
- Each group generates two samples: swap 1 is A@θ1, B@θ2, with target A; swap 2 is A@θ2, B@θ1, with target B. **Both samples query the direction θ1.**

## Offsets and alignment (same as SMS-WSJ scenario_map_fn; code checked in `repos/sms_wsj` commit f6b94bb)
- Offsets **follow the talker and do not change with azimuth**; they are the same for both swapped samples. The longer utterance starts at 0, and the shorter one at an integer drawn uniformly by the rng from [0, L_long − L_short] (8 kHz samples). Mixture length T = L_long; the two utterances overlap completely.
- For the 6-channel RIR of each position, rir_start = the first sample in each channel with |h| > 0.1·max|h|, taking the minimum (`get_rir_start_sample`). early RIR = h[: rir_start + 400] (50 ms), with the rest set to zero.
- After convolution, align with `extract_piece(x, offset − rir_start, T)`; both the full image and the early image are processed this way.

## Level
- In each swapped sample, each source is scaled so that **the active-frame RMS of the mic 0 early image = −30 dBFS**. Both swapped samples therefore have **SIR 0 dB** on the mic 0 early image.
- Active frames: the signal is cut into non-overlapping 256-point frames (32 ms); frames with power ≥ peak frame power − 40 dB count as active.
- The same gain is used for all 6 channels of that source, and for both the full image and the early image.
- Check of −30 dBFS (`scripts/sms_wsj/cv_dev93_level_stats.py`, srun 22013060, read-only):
  - for the 982 cv_dev93 mixture signals, computed on mic 0 with the same active-frame definition, the median active-frame RMS is −33.84 dBFS (p5 −34.35, p95 −32.72).
  - the median of each talker's early image is −35.18 dBFS.
  - −30 dBFS differs from the mixture median by 3.84 dB, within the 6 dB set in advance, so **−30 dBFS is kept**.

## Noise (adopted after checking SMS-WSJ)
- SMS-WSJ setting: 6-channel Gaussian white noise; SNR ~ U(20, 30) dB, with the reference power being the mean power of the clean mixture over all channels and all samples (`get_white_noise_for_signal`).
- This specification: each group uses one noise seed and one SNR (both recorded in the list), and only **one** noise realisation is generated; both swapped samples use **exactly the same** noise array. The noise is scaled with reference to the mean clean-mixture power of the two swapped samples. The actual SNR of an individual sample therefore deviates slightly around the group SNR (19.7–30.1 dB in the dry-run).

## Sample-rate conversion (the same resample_poly flow as C1b)
- The original 16 kHz SLURP audio is first downsampled to 8 kHz with `resample_poly(x, 1, 2)`, then convolved with the 8 kHz RIRs and mixed.
- Single-channel input for Qwen: take the mic 0 mixture signal, upsample to 16 kHz with `resample_poly(y, 2, 1)`, truncate to 2T.
- The separation front ends use the 6-channel 8 kHz mixture signal.

## Output (in the real run; the dry-run writes nothing)
- One directory `g{组号:03d}_s{1|2}/` per (group, swap), containing `obs_6ch_8k.wav`, `mic0_16k.wav`, `early_A_6ch_8k.wav`, `early_B_6ch_8k.wav` (all float32), plus a `manifest.jsonl`.
- Implementation: `scripts/sms_wsj/mix_slurp_v1.py`.

## Target direction instruction for C3 (frozen 2026-11-05)
Placed before the P2 prompt, as the first text segment after the audio; {theta} is θ1 rounded to whole degrees:
> The audio contains two overlapping speakers. Process only the speaker at azimuth {theta} degrees. Azimuth is measured from the array center, starting from the direction of microphone 0, counterclockwise positive, in the range -180 to 180 degrees.

- **Check of the positive direction** (cv_dev93 sample `0_4k6c0303_4k4c0319`):
  - the third coordinate is height (room height 3.073 m, microphone z between 1.550 and 1.563 m, source z = 1.557 m).
  - az = atan2(dy, dx) computed in the x-y plane, az_rel = wrap(az − az_mic0), implementation in `scripts/sms_wsj/sms_wsj_geometry_stats.py`.
  - the az of the 6 microphones are 125.1°, 184.9°, 244.9°, 304.8°, 4.9°, 64.8° in order, increasing counter-clockwise with the index.
  - Conclusion: under the premise that "x-y-z is right-handed with the z axis pointing up" (the usual convention for image-method RIRs, not stated explicitly in the SMS-WSJ code), a positive az_rel is counter-clockwise viewed from above, consistent with the instruction text.
- The two swapped samples have the same θ1, so the text is exactly the same; only the audio differs.
- The input of C3 is the **mic 0 single-channel** mixture signal; the audio itself contains no multichannel information that could be used to judge azimuth. This is by design of the C3 condition.
