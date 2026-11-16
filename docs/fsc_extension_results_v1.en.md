English translation of fsc_extension_results_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：fsc_extension_results_v1.md

# FSC extension v1: run records and results (2026-11-12)

> - Executed according to `docs/fsc_extension_plan_v1.md`; the freeze point is commit eaef16a (plan, mapping table `configs/fsc_mapping_v1.json`, scoring code `scripts/fsc/score_extension_v1.py`). After that, **neither the mapping table nor the scoring code was changed**.
> - Neither FSC official test nor SLURP test was used.
> - All jobs succeeded at the first attempt; nothing was rerun.
> - This file only collates results; no result file was changed.

## 1. Run records

| Step | Job | Status / time | Key points |
|---|---|---|---|
| Sampling and pairing | 22082964 (<CPU_PARTITION>) | COMPLETED 0:16 | see below |
| Reverberation | 22082992 | COMPLETED 1:22 | 1,240 recordings; 1,240 test_eval92 samples, all distinct; no clipping, highest peak 0.438 |
| Mixing | 22082993 | COMPLETED 1:57 | 720 samples; no sample point above 1 in either the observations or mic0 |
| Direct-path references | 22082994 (sms_wsj environment) | COMPLETED 2:39 | cv_dev93 check max difference 1.44e-8; early-image recomputation max difference 1.48e-8 |
| SpatialNet cv check | 22082995 (GPU) | COMPLETED 1:10 | PASS: direct SI-SDRi 25.93 dB; stream-selection accuracy 0.32 before polarity correction, 1.00 after |
| Separation | 22082996 (GPU) | COMPLETED 1:55 | 720 samples × 2 streams (MIX_DONE 720) |
| Stream selection | 22082997 | COMPLETED 2:00 | 720 samples / 1,440 files verified |
| Task lists | 22083468 | COMPLETED | clean / reverb / C0 1,240 each; spatial 720 for each of the two streams |
| Inference | 22083521–22083572 (52 GPU jobs) | 52 / 52 COMPLETED 0:0 | for each job, the item count, uniqueness of `item_id` and line-by-line JSON were checked with sacct and the output. The longest job was clean wt2, 16:24 |
| Scoring | 22085138 | COMPLETED 0:30 | `outputs/fsc_ext_v1/score/report.json` |
| Exploratory pilot recomputation | 22085139 | COMPLETED 0:08 | `outputs/fsc_ext_v1/pilot_exploratory/pilot_exploratory.json` |

- **Sampling and pairing**:
  - 1,240 single-talker recordings (248 sentence texts, 10 speakers; 951 female, 289 male). Of these, 125 recordings (25 sentence texts) have text identical to SLURP train and are flagged.
  - 281,484 candidate pairs; the integer program uses the top 20,000 by rank. **360 groups were feasible at once** (HiGHS Optimal), above the stopping line of 300 groups.
  - 90 groups in each of the four angle strata; the 360 test_eval92 samples are all distinct.
  - 529 recordings used, each at most 3 times. At most 90 groups per speaker (exactly the 25% cap). 60 male groups (lower limit 36 groups).
- **GPU usage**: inference 8.05 h plus SpatialNet 0.05 h, total **8.1 GPU hours**. Not above 12 h; no route was dropped.
- **Stream selection** (descriptive only, `outputs/fsc_ext_v1/selection/eval_snapshot_mix.json`):
  - mean two-talker direct SI-SDRi 16.55 dB;
  - stream-selection accuracy 0.949 overall, 0.985 excluding [0,15);
  - by stratum: [0,15) 0.839, [15,30) 0.972, [30,60) 0.994, [60,180] 0.989.
  - Both are lower than on SLURP test (18.29 dB, 0.980).

## 2. Primary comparison (audio LoRA − Whisper + T2-LoRA, target-any, paired; Bonferroni 3 comparisons)

Source: `outputs/fsc_ext_v1/score/report.json` → `primary`.

| Condition | n | Audio LoRA | Whisper+T2-LoRA | Difference | 95% interval | **98.33% interval** | Significant |
|---|---|---|---|---|---|---|---|
| clean | 1,240 recordings / 248 sentences | 0.909 | 0.923 | **−0.014** | [−0.043, +0.015] | **[−0.049, +0.020]** | No |
| reverb | 1,240 / 248 | 0.877 | 0.861 | **+0.017** | [−0.015, +0.048] | **[−0.021, +0.054]** | No |
| spatial (C5, equal weight) | 720 samples / 360 groups | 0.815 | 0.799 | **+0.017** | [−0.011, +0.044] | **[−0.021, +0.050]** | No |

- **None of the three is significant**: all 98.33% intervals cross 0, and all 95% intervals also cross 0.
- On FSC, **there is no evidence that the audio LoRA outperforms Whisper + T2-LoRA**; the point estimate for clean is in the opposite direction.
- Naturally weighted version of spatial (cv_dev93, descriptive only): −0.004 [−0.037, +0.029].
- The actual half-widths of the 98.33% intervals are 0.035 (clean), 0.037 (reverb) and 0.035 (spatial), slightly wider than the planned target of ±0.03, consistent with the estimate in plan §1: grouping by 248 sentence texts, with within-group correlation not equal to 0.
- **Text-overlap sensitivity** (removing the 25 sentences identical to SLURP train texts, leaving 1,115 recordings; descriptive only):
  - clean −0.016 [−0.048, +0.012];
  - reverb +0.014 [−0.020, +0.048].
  - The conclusion is unchanged.
- **Comparison with SLURP test**: all three are significant on SLURP test (+0.033, +0.055, +0.073); none is significant on FSC, and all point estimates are within ±0.02.

## 3. Descriptive results

### 3.1 Single-talker conditions and C0 (`report.json` → `conditions.clean / reverb / c0`)

| Condition / route | target-any | intent_exact | intent_family | action | object | location | Action unmappable | Object = none | Parse failures |
|---|---|---|---|---|---|---|---|---|---|
| clean original P2 | 0.848 | 0.557 | 0.665 | 0.781 | 0.759 | 0.853 | 0.188 | 0.159 | 1 |
| clean audio LoRA | 0.909 | 0.597 | 0.600 | 0.665 | 0.694 | 0.980 | 0.224 | 0.277 | 0 |
| clean original Whisper+T2 | 0.825 | 0.525 | 0.632 | 0.752 | 0.735 | 0.849 | 0.221 | 0.186 | 23 |
| clean Whisper+T2-LoRA | 0.923 | 0.582 | 0.588 | 0.693 | 0.677 | 0.972 | 0.190 | 0.149 | 0 |
| reverb original P2 | 0.815 | 0.518 | 0.616 | 0.727 | 0.727 | 0.841 | 0.223 | 0.191 | 1 |
| reverb audio LoRA | 0.877 | 0.565 | 0.570 | 0.632 | 0.675 | 0.969 | 0.238 | 0.287 | 0 |
| reverb original Whisper+T2 | 0.756 | 0.467 | 0.564 | 0.686 | 0.665 | 0.831 | 0.278 | 0.258 | 32 |
| reverb Whisper+T2-LoRA | 0.861 | 0.531 | 0.539 | 0.638 | 0.631 | 0.947 | 0.229 | 0.190 | 0 |
| C0 original T2 (reference transcripts) | 0.875 | 0.589 | 0.694 | 0.827 | 0.782 | 0.875 | 0.157 | 0.125 | 30 |
| C0 T2-LoRA (reference transcripts) | 0.956 | 0.613 | 0.613 | 0.714 | 0.694 | 1.000 | 0.157 | 0.149 | 0 |

- **Improvement over the respective baselines** (target-any, paired [95%]):

| Condition | Audio LoRA − original P2 | T2-LoRA − original Whisper+T2 |
|---|---|---|
| clean | +0.061 [+0.025, +0.097] | +0.098 [+0.056, +0.138] |
| reverb | +0.063 [+0.029, +0.098] | +0.105 [+0.067, +0.141] |
| C0 (T2-LoRA − T2) | — | +0.081 [+0.036, +0.129] |

  The improvements are far smaller than the roughly +0.25 to +0.28 on SLURP test.
- **Note that the target-any improvement comes mainly from the location slot**: the location accuracy of both LoRAs is about 0.95–1.00, versus about 0.83–0.88 for the baselines. **On the action and object slots, both LoRAs are below the baselines**, e.g. clean action 0.665 / 0.693 vs 0.781 / 0.752. intent_family (action + object) is also lower for the LoRAs.
- **Reason**: both LoRAs were fine-tuned on SLURP labels and tend to output SLURP-style intents and entities. After the frozen mapping, the proportions of unmappable actions and object = none are higher: object = none is 0.277 for the audio LoRA versus 0.159 for the baseline. intent_exact is still slightly above the baselines (0.597 / 0.582 vs 0.557 / 0.525).

### 3.2 Spatial condition and c6-real-v1 main conditions (`report.json` → `conditions.spatial`)

| Condition / abstention | Coverage (natural weighting) | target-any (equal weight): P2 / audio LoRA / WT2 / T2-LoRA | Interferer-only (equal weight): P2 / audio LoRA / WT2 / T2-LoRA | Audio − T2-LoRA target-any (equal weight [95%]) |
|---|---|---|---|---|
| C5 (θ1) | 1.000 | 0.743 / 0.815 / 0.701 / 0.799 | 0.064 / 0.083 / 0.067 / 0.092 | +0.017 [−0.011, +0.044] |
| 8 sectors / no abstention | 1.000 | 0.510 / 0.536 / 0.478 / 0.538 | 0.146 / 0.172 / 0.136 / 0.186 | −0.001 [−0.022, +0.018] |
| 8 sectors / abstention A | 0.826 | 0.780 / 0.817 / 0.728 / 0.827 | 0.052 / 0.052 / 0.059 / 0.069 | −0.010 [−0.044, +0.022] |
| 8 sectors / abstention B | 0.869 | 0.768 / 0.811 / 0.718 / 0.811 | 0.058 / 0.060 / 0.062 / 0.075 | 0.000 [−0.032, +0.030] |
| σ10 / no abstention | 1.000 | 0.662 / 0.724 / 0.623 / 0.717 | 0.141 / 0.160 / 0.134 / 0.170 | +0.007 [−0.018, +0.031] |
| σ10 / abstention A | 0.842 | 0.794 / 0.842 / 0.735 / 0.843 | 0.049 / 0.054 / 0.058 / 0.073 | −0.001 [−0.033, +0.029] |

- **Pair-flip accuracy** (C5): original P2 0.267, audio LoRA 0.328, original WT2 0.222, T2-LoRA 0.281.
- **Interferer-only is much higher than on SLURP** (C5 about 0.06–0.09; SLURP 0.003–0.018). There are interferer hits even with correct routing (31–46 / 683 per route), suggesting that one issue with FSC is that the two talkers' triples share many values. FSC has only 31 triples; location and action are often the same, so there are fewer distinguishable values; this was not checked further. **The post-hoc stratification (Section 6.1) does not support this explanation.**
- **Interferer risk** (natural weighting, paired [95%]; descriptive only):
  - audio LoRA − original P2: 8 sectors no abstention +0.003 [−0.021, +0.027], σ10 no abstention −0.000 [−0.024, +0.023];
  - T2-LoRA − original WT2: 8 sectors no abstention +0.044 [+0.017, +0.071], σ10 no abstention +0.038 [+0.012, +0.063].
  - Numerically, the increase on FSC appears mainly on the T2-LoRA side; the naturally weighted interferer-only difference audio LoRA − T2-LoRA is about −0.036 to −0.044 on C5 and under each main condition (e.g. C5 −0.036 [−0.067, −0.005]).
  - **Interpretation (revised 2026-11-13)**: 92–98% of interferer hits under correct routing come from default values filled in by the intent mapping (Section 6.3). The difference in interferer risk between the two LoRAs on FSC therefore **mainly reflects the mapping and cannot be interpreted as a difference in model behaviour**; the values above are descriptive only.
- **Split by actual routing** (no abstention; observations / target-value hits / interferer-value hits):

| Condition / routing | Original P2 | Audio LoRA | Original WT2 | T2-LoRA |
|---|---|---|---|---|
| C5 correct (683) | 963 / 36 | 1,019 / 39 | 902 / 35 | 973 / 48 |
| C5 incorrect (37) | 10 / 24 | 12 / 32 | 10 / 23 | 9 / 29 |
| 8 sectors correct (601) | 856 / 35 | 897 / 34 | 786 / 30 | 858 / 43 |
| 8 sectors incorrect (119) | 15 / 131 | 19 / 148 | 15 / 122 | 16 / 146 |
| σ10 correct (6,071) | 8,536 / 321 | 9,000 / 330 | 7,950 / 307 | 8,618 / 415 |
| σ10 incorrect (1,129) | 125 / 1,337 | 139 / 1,431 | 118 / 1,223 | 153 / 1,422 |

## 4. Recomputing the 64 pilot utterances with the new mapping (**exploratory, separate from the extension results**)

Source: `outputs/fsc_ext_v1/pilot_exploratory/pilot_exploratory.json`, sha256 `0ff1cc6c…7ad7`. The table below is "target-any / intent_exact"; the pilot has only 3 routes.

| Condition | Original Qwen | Audio LoRA | Whisper+T2-LoRA | Audio − T2-LoRA (target-any [95%], grouped by text) |
|---|---|---|---|---|
| clean | 0.859 / 0.547 | 0.953 / 0.672 | 0.953 / 0.609 | 0.000 [−0.063, +0.062] |
| reverb | 0.812 / 0.531 | 0.938 / 0.625 | 0.906 / 0.562 | +0.031 [−0.048, +0.101] |
| spatial | 0.766 / 0.484 | 0.906 / 0.562 | 0.875 / 0.547 | +0.031 [−0.047, +0.108] |
| direction offset | 0.531 / 0.344 | 0.625 / 0.406 | 0.578 / 0.375 | +0.047 [−0.016, +0.121] |

- The pilot reverb used cv_dev93 RIRs, and spatial looked only at the selected stream with no inference on the other stream, so it is not directly comparable with the extension results.
- The pilot has only 64 utterances and serves only as an exploratory reference.

## 5. Limitations
- **Dependence on the mapping**: the FSC label scheme differs from SLURP's, and scoring depends on the frozen mapping table.
  - Both LoRAs are below the baselines on the action and object slots, possibly partly because the SLURP-style entities they output cannot be mapped to FSC values under the frozen rules. For example, lights is accepted only in the plural, and music is not an object keyword.
  - The improvement of "LoRA over baseline" on FSC may therefore be under-estimated.
  - But both LoRAs use the same mapping, so the effect on the primary comparison (audio − T2-LoRA) is symmetric.
- **Corpus size**: FSC valid has only 10 speakers and 248 sentence texts, with highly repetitive sentence patterns. The interval half-width is about 0.035, slightly wider than on SLURP.
- **The drafter had been exposed to pilot outputs**: the author of the mapping table had seen part of the pilot outputs (stated in the plan). The mapping rules are based only on label definitions and FSC train frequencies.
- **Front-end conditions**: the spatial condition is still a simulated mixture, with RIRs from SMS-WSJ test_eval92, the same kind as for SLURP test; SpatialNet separation and stream selection are both worse on FSC than on SLURP.
- **Same checkpoints**: both LoRAs use the same checkpoints selected on SLURP, not adjusted on FSC.

## Appendix: sha256 of source files
| File | sha256 |
|---|---|
| `outputs/fsc_ext_v1/score/report.json` | `dc694398b0ff00f22b4e7eccc6aa655b4e1c63da65fe089e21521299193218e5` |
| `outputs/fsc_ext_v1/score_inputs.json` | `535ec633d27904d57205f6d45961ac764b09b2b54cefa04eaede8ee9e63152d5` |
| `outputs/fsc_ext_v1/pilot_exploratory/pilot_exploratory.json` | `0ff1cc6c9d9adeb0875a6238532407da9d39a3c033625fdd2f7b34dcc7367ad7` |
| `configs/fsc_ext_v1_list.json` | `ea4accd00a982e8c704bcff03c9de1c79dd9600a67d53c327d94868abcc63678` |
| `configs/fsc_mapping_v1.json` | `7a385e29c67bf9ac57ac034aa4149583b0d5432179cea71e9e1363d48f2f0ced` |
| `scripts/fsc/score_extension_v1.py` | `24c48ff83d0323660dfd205169be52676f0d5cc9a0827206f57b1be122cf2a8b` |
| `outputs/fsc_ext_v1/selection/eval_snapshot_mix.json` | `594cdb8845c4d8996486a5512c760bcabd5cd3ef948baa1ba6dd9b3480e084b8` |
| `outputs/fsc_ext_v1/tasks/clean.jsonl` | `540836bfa9dadda4b08797d13df3428fd0aeda3f1ad5419e851ddc1bedc3673c` |
| `outputs/fsc_ext_v1/tasks/reverb.jsonl` | `11999b7dd382a1ea822c89a7710c87e5b5020d936db204c02396926ca6195123` |
| `outputs/fsc_ext_v1/tasks/c0.jsonl` | `b35b21fa4aab6b356df489c9ef079c1f112f9b853fb24b9ffc82af41f7623235` |
| `outputs/fsc_ext_v1/tasks/spatial_selected.jsonl` | `568e61ae599ada01d399b728534d24bad019a86ff0498eea8758051563467d39` |
| `outputs/fsc_ext_v1/tasks/spatial_other.jsonl` | `d234be9ac57a43e50c4c635c191d0ce3a22f576a1c9d4225f20212caa07db2f8` |
| `data/fsc_ext_v1/reverb16k/manifest.jsonl` | `7317f96abd63264b82f8727a6718204df4685e90e0ac552ed9ab79217728b228` |
| `data/fsc_ext_v1/mix/manifest.jsonl` | `d2fb10fc28311499efd2b98c2cfe902fbca7f6702db65cffb16deff461c78ab1` |

## 6. Post-hoc exploratory analyses (**post-hoc exploratory, no verdicts**; 2026-11-13)

> - Read-only CPU analysis; no job was submitted.
> - Computed on the frozen v1 outputs with the frozen scoring function (`score_extension_v1.mix_row`, unchanged).
> - Script `scripts/fsc/explore_overlap_strata.py`, output `outputs/fsc_ext_v1/explore_overlap/overlap_strata.json` (sha256 `9ff2225790604b266b2f5fd520235eeff83108f6e66e5adb98df1611a1921ffd`).
> - Only sample counts and proportions are reported, **with no significance judgements**; no conclusion of Section 2 is changed.

### 6.1 Stratified by the overlap between the two talkers' triples (C5: query direction θ1, selected stream, equal weight)

Overlap = the number of slots with the same label in the target and interferer triples (action, object, location; `none` = `none` also counts as the same). The pairing rules require at least some distinguishable value, so at most 2 slots can be the same.

| Same slots | Samples | target-any: original P2 / audio LoRA / T2-LoRA | Interferer-only: original P2 / audio LoRA / T2-LoRA |
|---|---|---|---|
| 0 | 330 | 0.809 / 0.873 / 0.864 | 0.085 / 0.091 / 0.133 |
| 1 | 300 | 0.707 / 0.770 / 0.760 | 0.037 / 0.070 / 0.057 |
| 2 | 90 | 0.622 / 0.756 / 0.689 | 0.078 / 0.100 / 0.056 |

| Same slot(s) | Samples | target-any: P2 / audio / T2-LoRA | Interferer-only: P2 / audio / T2-LoRA |
|---|---|---|---|
| none | 330 | 0.809 / 0.873 / 0.864 | 0.085 / 0.091 / 0.133 |
| location only | 224 | 0.692 / 0.750 / 0.728 | 0.045 / 0.071 / 0.062 |
| action only | 40 | 0.775 / 0.850 / 0.875 | 0.025 / 0.075 / 0.050 |
| object only | 36 | 0.722 / 0.806 / 0.833 | 0.000 / 0.056 / 0.028 |
| object + location | 38 | 0.684 / 0.658 / 0.526 | 0.105 / 0.105 / 0.053 |
| action + object | 30 | 0.500 / 0.867 / 0.833 | 0.067 / 0.100 / 0.033 |
| action + location | 22 | 0.682 / 0.773 / 0.773 | 0.045 / 0.091 / 0.091 |
- Values for original Whisper + T2 are in the json.
- **Observations** (descriptive only):
  - target-any decreases as the number of same slots increases. One reason is that there are fewer distinguishable values and therefore fewer chances to hit.
  - Interferer-only does **not** rise with overlap. T2-LoRA is in fact highest in the stratum with no overlap at all (0.133). The "FSC two-talker triples share many values" mentioned in Section 3.2 therefore **cannot explain** the higher interferer hits on FSC.
  - Some strata have only 22–40 samples, so the numbers fluctuate a lot.

### 6.2 Interferer-value hits split by actual routing correct / incorrect: FSC and SLURP test side by side (no abstention)

Sources: FSC `outputs/fsc_ext_v1/score/report.json`; SLURP `outputs/slurp_test_v1/score/checks_p2/checks.json`, `checks_t2/checks.json`. Each cell is "observations / interferer-value hits (hits per observation)".

| Condition | Routing | Dataset | Original P2 | Audio LoRA | Original Whisper+T2 | Whisper+T2-LoRA |
|---|---|---|---|---|---|---|
| C5 | correct | FSC | 683 / 36 (0.053) | 683 / 39 (0.057) | 683 / 35 (0.051) | 683 / 48 (0.070) |
| C5 | correct | SLURP | 392 / 0 (0.000) | 392 / 3 (0.008) | 392 / 0 (0.000) | 392 / 3 (0.008) |
| C5 | incorrect | FSC | 37 / 24 (0.649) | 37 / 32 (0.865) | 37 / 23 (0.622) | 37 / 29 (0.784) |
| C5 | incorrect | SLURP | 8 / 1 (0.125) | 8 / 3 (0.375) | 8 / 1 (0.125) | 8 / 4 (0.500) |
| 8 sectors | correct | FSC | 601 / 35 (0.058) | 601 / 34 (0.057) | 601 / 30 (0.050) | 601 / 43 (0.072) |
| 8 sectors | correct | SLURP | 337 / 0 (0.000) | 337 / 3 (0.009) | 337 / 0 (0.000) | 337 / 3 (0.009) |
| 8 sectors | incorrect | FSC | 119 / 131 (1.101) | 119 / 148 (1.244) | 119 / 122 (1.025) | 119 / 146 (1.227) |
| 8 sectors | incorrect | SLURP | 63 / 26 (0.413) | 63 / 44 (0.698) | 63 / 27 (0.429) | 63 / 42 (0.667) |
| σ10 (× 10) | correct | FSC | 6,071 / 321 (0.053) | 6,071 / 330 (0.054) | 6,071 / 307 (0.051) | 6,071 / 415 (0.068) |
| σ10 (× 10) | correct | SLURP | 3,420 / 0 (0.000) | 3,420 / 27 (0.008) | 3,420 / 0 (0.000) | 3,420 / 27 (0.008) |
| σ10 (× 10) | incorrect | FSC | 1,129 / 1,337 (1.184) | 1,129 / 1,431 (1.267) | 1,129 / 1,223 (1.083) | 1,129 / 1,422 (1.260) |
| σ10 (× 10) | incorrect | SLURP | 580 / 270 (0.466) | 580 / 464 (0.800) | 580 / 239 (0.412) | 580 / 425 (0.733) |

- **Observations** (descriptive only):
  - **With correct routing**, all four routes on FSC have about 0.05–0.07 interferer-value hits per observation, versus 0–0.009 on SLURP.
  - **With incorrect routing**, the LoRAs are above their respective baselines on both datasets.
  - Interferer hits under correct routing on FSC may come from the following causes, **all unverified**:
    1. residual crosstalk after separation (FSC separation SI-SDRi is lower);
    2. default values of the frozen mapping. For example, the intent-default object lights happens to equal the interferer's object, which is fairly common among FSC's closed set of 31 triples;
    3. FSC utterances are short and have a small vocabulary.
  - The hits of T2-LoRA under correct routing on FSC (0.068–0.072) are higher than those of the other three routes (0.050–0.058); this is one of the main sources of the negative "audio LoRA − T2-LoRA interferer-only" on FSC. The cause was not investigated.

### 6.3 Sources of interferer-value hits under correct routing (**post-hoc exploratory, no verdicts**; 2026-11-13)

> - Script `scripts/fsc/explore_hit_sources.py`, output `outputs/fsc_ext_v1/explore_overlap/hit_sources.json` (sha256 `bf3a8f61c2c484af5d2c1655ce8406932376c22c01a03bdd43c7bf4bf4b5ca7c`).
> - The script re-traces the source of each predicted value under the frozen rules, and asserts item by item that the traced result is exactly the frozen `predict()` triple and that the hit count equals `mix_row` (all pass). Neither mapping nor scoring was changed.
> - Scope: C5 (query direction θ1, selected stream), the 683 samples where the selected stream = the target reference stream.

Each hit is classified by "the source of the predicted value in that slot". The action can only come from the intent (or be an action determined by the object); the location can only come from entity values (there is no default location).

| Route | Total hits (per observation) | Intent-default object | Action mapped from intent | Entity values from model output (object / location / action determined by object) |
|---|---|---|---|---|
| Original P2 | 36 (0.053) | 20 | 13 | 3 |
| Audio LoRA | 39 (0.057) | 12 | 23 | 4 |
| Original Whisper+T2 | 35 (0.051) | 21 | 12 | 2 |
| Whisper+T2-LoRA | 48 (0.070) | 27 | 20 | 1 |

- **Observations** (descriptive only):
  - In all four routes, **92–98% of hits come from values mapped from the intent** (intent-default object + action mapped from intent); only 1–4 come from entity values output by the model.
  - The additional hits of T2-LoRA over the audio LoRA come mainly from **intent-default objects** (27 vs 12).
  - In other words, interferer hits under correct routing are mainly not the model extracting entity values from the interferer's speech, but the mapping intent → FSC action or default object happening to equal a value in the interferer's triple. For example, if the target is (activate, lamp) and the interferer is (activate, lights), and the model outputs iot_hue_lighton, the default object lights is mapped out.

### 6.4 Interferer hits under correct routing versus separation SI-SDRi (**post-hoc exploratory, no verdicts**)

The 683 correctly routed samples are split into three equal parts by the direct SI-SDRi of the target stream (`sisdri_target` in `per_sample.jsonl`).

| SI-SDRi tertile | Samples | Range (dB) | Interferer hits per observation: P2 / audio LoRA / WT2 / T2-LoRA | Interferer-only: P2 / audio LoRA / WT2 / T2-LoRA |
|---|---|---|---|---|
| low | 228 | −9.1 to 15.5 | 0.083 / 0.070 / 0.066 / 0.066 | 0.070 / 0.070 / 0.057 / 0.061 |
| middle | 227 | 15.5 to 19.6 | 0.031 / 0.053 / 0.044 / 0.079 | 0.026 / 0.048 / 0.040 / 0.075 |
| high | 228 | 19.6 to 30.3 | 0.044 / 0.048 / 0.044 / 0.066 | 0.039 / 0.048 / 0.044 / 0.066 |

- **Observations** (descriptive only):
  - Original P2 has the most hits in the low-SI-SDRi stratum; the other routes show no consistent monotonic relationship.
  - T2-LoRA is about 0.066–0.079 in all three strata.
  - Separation quality cannot explain interferer hits under correct routing, consistent with the observation in Section 6.3 that they "come mainly from mapped values".
