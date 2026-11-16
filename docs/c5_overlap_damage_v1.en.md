English translation of c5_overlap_damage_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c5_overlap_damage_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C5 overlap damage diagnosis v1 (c5-overlap-damage-v1; 2026-11-05; read-only, written before computation; descriptive only, **correlation not written as causation**, no new audio)

## A. Breakdown of Whisper errors
- Object: W_C4direct (direct reference) and W_C5 (C5 target stream) of job 22016221, 200 samples each. The reference is the target sentence; both sides use normalize_v2.
- Levenshtein alignment (cost 1 for substitution, deletion and insertion; on ties during backtracking, prefer substitution, then deletion, then insertion) gives the numbers of substitutions S, deletions D and insertions I. Check that S + D + I agrees with the result of `slu_normalize.wer_counts`.
- Report: the totals of S, D and I for each group and their proportion of the number of reference words.
- Item-by-item comparison: ΔWER = WER(W_C5) − WER(W_C4direct); count the items where C5 is worse, the same or better, and the sums of ΔS, ΔD and ΔI.
- List the 10 samples of W_C5 with the highest WER: reference sentence, W_C5 transcript, S/D/I, and the main error type. Rules for the main error type:
  - if the WER of the transcript against the interferer sentence is lower than against the target sentence, record "leaning to the interferer";
  - otherwise take the largest of S, D, I (on ties, in the order substitution, deletion, insertion);
  - if the number of insertions exceeds the number of reference words, additionally mark "massive insertion".

## B. Overlap-frame diagnosis (C5 target stream, 8 kHz)
- Frames: non-overlapping 256-point frames (32 ms). Active frames are determined separately with the direct references of the target and the interferer, by the same rule as mix_v1: frame power ≥ the peak frame power of that reference − 40 dB.
  - frames where **only the target speaks**: target active, interferer inactive;
  - **two-talker overlap** frames: both active;
  - other frames are not counted.
- Decomposition: the same as separation-quality-diag-v1, least squares after zero-mean, y = e_t + e_i + e_r, where e_t = a·d_T and e_i = b·d_I.
- For each class of frames, sum the energy within frames and compute:
  - target vs interference: 10·log10(Σ‖e_t‖² / Σ‖e_i‖²);
  - target vs residual: 10·log10(Σ‖e_t‖² / Σ‖e_r‖²);
  - target vs everything else: 10·log10(Σ‖e_t‖² / Σ‖y − e_t‖²), equivalent to a segmental SDR.
- Report: the median and quartiles of these metrics for both frame classes; the proportion of two-talker overlap frames among all active frames; comparison between the two groups split by whether C5 hits (target-any).
- Implementation: `scripts/sms_wsj/c5_overlap_damage.py` (contains both A and B; run with srun on a CPU node), results written to `outputs/c5_overlap_damage_v1/`.

## Results (2026-11-05; descriptive only, **not causal**)
- Run record: srun 22016733 (<CPU_PARTITION>, tfgridnet environment), script commit `eb85b2e`. Results in `outputs/c5_overlap_damage_v1/{overlap_damage.json,per_sample.jsonl}`. No new audio.

### A. Breakdown of Whisper errors (1,162 reference words in total)
| | Substitutions S | Deletions D | Insertions I | WER |
|---|---|---|---|---|
| W_C4direct | 62 (0.053) | 16 (0.014) | 12 (0.010) | 0.077 |
| W_C5 | 353 (0.304) | 110 (0.095) | 62 (0.053) | 0.452 |
| Difference (C5 − C4direct) | +291 | +94 | +50 | +0.375 |

- Item-by-item comparison (200 items): C5 worse for 130, the same for 65, C5 better for 5.
- Mean WER grouped by whether C5 hits: W_C5 hit 0.209, miss 0.720; W_C4direct in the same two groups 0.046 and 0.140.

### The 10 samples of W_C5 with the highest WER
| key | Reference sentence | W_C5 transcript | S / D / I | WER against target / interferer sentence | Main error type (by the rules) |
|---|---|---|---|---|---|
| g008_s2 | appointments this wednesday | Is the meeting scheduled for tomorrow? | 3/0/3 | 2.00 / **0.00** | leaning to the interferer |
| g017_s2 | indie | Thank you. | 1/0/1 | 2.00 / 1.00 | leaning to the interferer |
| g069_s1 | indie | Thank you. | 1/0/1 | 2.00 / 1.00 | leaning to the interferer |
| g081_s1 | indie | Thank you. | 1/0/1 | 2.00 / 1.00 | leaning to the interferer |
| g015_s2 | who is arnold schwarzenegger | It's on your side of the mega. | 4/0/3 | 1.75 / 1.17 | leaning to the interferer |
| g012_s2 | find recipe chicken casserole | Find rest of the children to grow. | 3/0/3 | 1.50 / 1.40 | leaning to the interferer |
| g037_s1 | who is arnold schwarzenegger | is Arden was sizing his makeup. | 4/0/2 | 1.50 / 1.00 | leaning to the interferer |
| g056_s1 | closest chipotle | Close this checkbook. | 2/0/1 | 1.50 / 1.00 | leaning to the interferer |
| g074_s1 | add pencil to school supplies list | I should thank you on the new program release. | 6/0/3 | 1.50 / 1.00 | leaning to the interferer |
| g074_s2 | put pencil on a new grocery list | I think it's better than you can run with it. | 7/0/3 | 1.43 / 1.67 | substitution |

- **Note on the rule**: the condition for "leaning to the interferer" in the rule is only "WER against the interferer sentence lower than WER against the target sentence". Of these 10 samples, only **g008_s2** was really transcribed as the interferer sentence (WER 0 against the interferer sentence). The others labelled "leaning to the interferer" also have WER ≥ 1.0 against the interferer sentence and in fact resemble neither sentence, e.g. "Thank you.", "Close this checkbook.". The rule is kept unchanged; this is only an explanation. C5 misses all 10 of these samples.

### B. Overlap-frame diagnosis (median [P25, P75], in dB)
| | All 200 | C5 hit (79) | C5 miss (121) |
|---|---|---|---|
| Proportion of two-talker overlap frames among active frames | 0.72 [0.65, 0.81] | 0.71 | 0.73 |
| Target-only frames: target/interference | 67.4 [53.3, 82.4] (155 samples have such frames) | 70.2 | 65.6 |
| Target-only frames: target/residual | 6.67 [3.69, 8.80] | 7.08 | 6.45 |
| Two-talker overlap frames: target/interference | **29.6** [20.1, 39.5] | 33.6 | 27.9 |
| Two-talker overlap frames: target/residual | **5.77** [3.68, 7.70] | 6.54 | 5.59 |
| Two-talker overlap frames: segmental SDR | 5.66 [3.34, 7.70] | 6.43 | 5.56 |

- Description:
  - In overlap frames, residual interference is clearly higher than in target-only frames (median target/interference drops from 67 dB to 30 dB), but still much smaller than the residual. Target/residual differs by only about 0.9 dB between the two frame classes.
  - The C5 hit group has somewhat higher target/interference and target/residual on overlap frames than the miss group, by about 5.8 dB and 0.9 dB respectively.
  - In target-only frames the target/interference values are very high, as expected by definition: these frames have no direct-sound energy from the interferer in the first place.
  - These are only correlations and do not show which factor causes downstream failure.
