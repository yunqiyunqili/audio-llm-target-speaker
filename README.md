**English** | [中文](README_zh.md)

# Spatial-audio spoken language understanding: audio LLM vs. cascaded ASR (SLURP + FSC)

## Problem
When two people talk at the same time and only the approximate direction of the target speaker is known, how should the target's speech be passed to a large model for spoken language understanding (intent + entities)? This project obtains the target speech with multichannel spatial separation (SpatialNet) plus direction-based stream selection, and then compares two routes:
- **Audio route**: the separated target speech goes directly into Qwen2.5-Omni-7B, fine-tuned with an audio LoRA;
- **Cascaded route**: Whisper large-v3 transcribes the speech, and Qwen fine-tuned with a text LoRA (T2-LoRA) performs SLU.

The pre-specified primary comparison is **Audio LoRA − Whisper + T2-LoRA** (target-any, paired), tested on three conditions: single-talker clean (C1), single-talker reverberant (C1-R), and two-talker after separation and stream selection (C5). Bonferroni correction is applied (significant only if the lower bound of the 98.33% interval is > 0).

## Conclusions
- **SLURP test** (run once, under a frozen protocol): all three primary comparisons are significant; the Audio LoRA outperforms Whisper + T2-LoRA, but by a small margin (+0.03 to +0.07). The lower bound for C1 is close to 0; the natural-angle-weighted version of C5 crosses 0 (descriptive only).
- **FSC cross-dataset check**: **the primary comparison did not replicate**. None of the three comparisons is significant; point estimates lie between −0.014 and +0.017. Possible reasons (under-estimation by the label mapping, FSC being closed-domain and simple, a weaker front end) are **all unverified**.
- Relative to their own baselines, both LoRAs improve by about +0.25 to +0.28 on SLURP, but only by +0.06 to +0.10 on FSC.
- **Interferer risk**: on the realistic system path (simulated coarse direction / direction noise), the "interferer only" rate of both LoRAs on SLURP is about +0.02 higher than their own baselines; with abstention it drops to about +0.01. On FSC, 92–98% of interferer hits on correctly routed samples come from default values filled in by the intent mapping; the difference in interferer risk between the two LoRAs on FSC therefore mainly reflects the mapping and cannot be interpreted as a difference in model behaviour.
- All spatial conditions are simulated mixtures. **These results are not extrapolated to other datasets or to real recordings.**

## System pipeline
```
6-channel mixture (8 kHz)
 → SpatialNet-small ep100 separation (two streams)
 → polarity correction (zero-lag mic0 correlation)
 → PHAT direction estimation (one DOA per stream)
 → stream selection: the stream closest to the query direction (v3)
 → abstention (optional): A = margin between the two stream directions < 17°; B = both streams fall in the query sector
 → selected stream: resampled to 16 kHz, level −30 dBFS
 → audio route: Qwen2.5-Omni + P2 (original / audio LoRA)
   cascaded route: Whisper large-v3 → Qwen + T2 (original / T2-LoRA)
 → JSON (transcript, intent, entities) → slu-score-v2 / binding-score-v1
```

## Main results (Audio LoRA − Whisper + T2-LoRA, target-any)

| Dataset / condition | n | Audio LoRA | Whisper+T2-LoRA | Difference | 98.33% interval | Significant |
|---|---|---|---|---|---|---|
| SLURP test C1 | 1,183 | 0.848 | 0.815 | +0.033 | [+0.006, +0.060] | Yes |
| SLURP test C1-R | 1,183 | 0.806 | 0.751 | +0.055 | [+0.025, +0.086] | Yes |
| SLURP test C5 | 400 | 0.755 | 0.683 | +0.073 | [+0.025, +0.120] | Yes |
| FSC clean | 1,240 | 0.909 | 0.923 | −0.014 | [−0.049, +0.020] | **No (not replicated)** |
| FSC reverb | 1,240 | 0.877 | 0.861 | +0.017 | [−0.021, +0.054] | **No (not replicated)** |
| FSC spatial (C5) | 720 | 0.815 | 0.799 | +0.017 | [−0.021, +0.050] | **No (not replicated)** |

Both LoRAs relative to their own baselines (paired target-any difference):

| Condition | Audio LoRA − original P2 | T2-LoRA − original Whisper+T2 |
|---|---|---|
| SLURP C1 | +0.281 | +0.263 |
| SLURP C1-R | +0.276 | +0.252 |
| SLURP C5 | +0.260 | +0.258 |
| FSC clean | +0.061 | +0.098 |
| FSC reverb | +0.063 | +0.105 |

Figures (`docs/figures/`, generated from the result files by `scripts/figures/make_figures.py`):
- `fig1_error_ladder.png`: error decomposition ladder;
- `fig2_risk_coverage.png`: risk–coverage;
- `fig3_forest.png`: forest plot of the primary comparison (devel / test / FSC).

## Limitations
- All spatial conditions are close-talk speech convolved with SMS-WSJ simulated RIRs plus white noise; there are no real-room or real-array recordings. Only two talkers at SIR 0 dB were tested.
- The query direction is the ground truth in C5 and a simulated sector / noise in c6-real; there is no real user input.
- Each LoRA was trained with a single seed and a single checkpoint; only one ASR system was used (Whisper large-v3, not fine-tuned).
- Only "hard selection of one stream" is implemented. Neither the fusion method M3 nor reliability-based stream selection M2 is implemented.
- On C3 (unseparated mixture + direction instruction), the Audio LoRA is more likely to output the interferer's entities: 0.128 → 0.195.
- FSC scoring depends on a frozen SLURP → FSC mapping table; the author of the mapping table had seen part of the pilot outputs (the rules are based only on label definitions and FSC train frequencies).

## Document index
| File | Content |
|---|---|
| `docs/slurp_test_report_v1.md` | **External report (paper draft)**: setup, main results, FSC cross-dataset check, risk, limitations, future work |
| `docs/slurp_test_freeze_v1.md` | SLURP test freeze protocol, approval decisions, S0–S5 run records (job IDs, commits, sha256) |
| `docs/fsc_extension_plan_v1.md` | FSC extension plan and frozen mapping table (basis and counts for every rule) |
| `docs/fsc_extension_results_v1.md` | FSC extension run records, results, post-hoc exploratory analyses |
| `docs/lora_slu_plan_v2.md` / `docs/lora_v2_runs.md` | LoRA training plan and all devel-stage run records |
| `docs/mix_spec_v1.md` | Two-talker mixture construction rules (mix_v1) |
| `docs/c6_real_v1.md` | Realistic direction-uncertainty evaluation (sector, noise, abstention A/B) |
| `docs/spatialnet_final_v1.md` | SpatialNet ep100 final checkpoint and devel evaluation |
| `docs/binding_score_v1.md` / `docs/scoring_rules.md` | Scoring definitions |
| `docs/fsc_pilot_plan.md` | FSC acquisition, audit and the 64-utterance pilot |
| `docs/plan_update_v2.md` | Main-line revision and M1/M2/M3 definitions |

## Reproduction
**Environments** (`envs/` → `<SHARED_ROOT>/envs`):
- `qwen_omni`: torch 2.6.0+cu124, transformers 4.52.4; for the original Qwen and Whisper;
- `qwen_omni_lora`: the same plus peft 0.15.2; for the LoRAs and most CPU scoring;
- `tfgridnet`: for SpatialNet inference, stream selection and plotting (matplotlib);
- `sms_wsj`: rir_generator 0.3.0; for direct-path references.
- Slurm: account <SLURM_ACCOUNT>; GPU jobs use <GPU_PARTITION>, 1×A100, 4 CPUs, 48 GB, 20 minutes. The job wrapper is `scripts/lora/slurp_test_job.sbatch`, which refuses to run on a dirty working tree; the environment variable `EVAL_SPLIT` is test / fsc_ext.

**Script order (SLURP test)**:
1. Split: `scripts/slurp/make_slurp_split_v2.py` → `configs/slurp_split_v2_seed2026.json` and `data/slurp/manifests/split_v2_seed2026/*`;
2. Training:
   - reverberant training data: `scripts/lora/reverb_single.py --split train`;
   - Audio LoRA: `scripts/lora/train_audio_lora.py`;
   - Whisper transcription: `scripts/lora/baseline_audio.py --mode whisper --items train_all`;
   - T2-LoRA: `scripts/lora/train_step2_lora.py`;
3. Test construction:
   - `scripts/slurp/make_mix_list_v1.py --v2 --split test --n 200`;
   - `scripts/sms_wsj/mix_slurp_v1.py`;
   - `scripts/lora/reverb_single.py --split test`;
   - `scripts/sms_wsj/direct_refs_mix_v1.py`;
4. Front end: `scripts/sms_wsj/frontend_snapshot.py --mode cv/mix`; `scripts/sms_wsj/select_final.py`;
5. Inference:
   - `scripts/lora/baseline_audio.py` (P2, Whisper + T2);
   - `scripts/lora/eval_audio_lora_ckpt.py`;
   - `scripts/lora/eval_step2_lora.py`;
   - `scripts/qwen/qwen_omni_c0_devel428.py --split test`;
   - `scripts/qwen/qwen_omni_c3_p2.py`;
   - `scripts/lora/eval_audio_lora_mix.py`;
   - `scripts/qwen/qwen_omni_snapshot.py`;
   - merging shards: `scripts/lora/merge_test_parts.py`;
6. Scoring: `scripts/qwen/eval_final_routes.py`; `scripts/lora/eval_final_lora_checks.py`; `scripts/lora/score_slurp_test_v1.py`.

All job IDs and commands are in `docs/slurp_test_freeze_v1.md`; `scripts/lora/slurp_test_s0_check.py` is the devel equivalence check.

**FSC**:
1. `scripts/fsc/ext_build_list.py`;
2. `ext_make_reverb.py`;
3. mixing / separation / stream selection reuse the scripts above with `EVAL_SPLIT=fsc_ext`;
4. `ext_tasks.py`;
5. `ext_batch.py`;
6. `score_extension_v1.py`.

**Key file sha256**:
| File | sha256 |
|---|---|
| Qwen2.5-Omni-7B | HF commit `ae9e1690` |
| Whisper large-v3 | HF revision `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| SpatialNet ep100 `best.pt` | `21caf0fc3b08cfc3b9860efbfeff9a756acecbe36101620378e47891edd76c10` |
| Audio LoRA `lora_v2/step1/lr1e-4/ckpt_step440/adapter_model.safetensors` | `66d6a51d69a5d86f196819cff24ba49aa817aed92055460dd837bef6d0bed0cf` |
| T2-LoRA `lora_v2/step2/lr1e-4/ckpt_step660/adapter_model.safetensors` | `904039c1763676bad4f41b2a45c033e1891bf1e2ebdad4a7add0f50bacb4c68b` |
| SLURP annotations `data/slurp/annotations/8eb1654/test.jsonl` | `fe3449af69b42fda7163345482556066c35a80f7e552a6d67e212a0e2f0783cc` |
| `configs/slurp_split_v2_seed2026.json` | original run version `eed16990a8f284c59393a7fed6fba2ec46b5dd2c3c215d1cb08e2f217a091d8c`; **public version** `2ff8e1901d884d320ff276d578b0d2ba4858559888c35790d2a9ed28fb871ccc` (differs only in path fields, which are replaced by placeholders) |
| `configs/mix_v1_test_v1.json` | original run version `26c0fb57e79fa4442526c824cd1d2c79e558af727dc1c11f41e4dea26b02695f`; **public version** `22283f4cb3fc09a20f32a7a5775a1689f10bee7374a5042ed49c3ee8439f91a5` (differs only in path fields, which are replaced by placeholders) |
| `configs/slurp_test_v1_s1_sha256.txt` (all test input files) | original run version `d6475d2e27fc77cd466176759d9baba57a4db6ef4eca868bc09e0395241cd972`; **public version** `ad8549a2b8a6c1284fdac60a27c2f4929d385651da6cb118f8f09fdc2915d342` (differs only in path fields, which are replaced by placeholders) |
| SLURP test report `outputs/slurp_test_v1/score/report/report.json` | `e457aa3cbdd16a6ec68bfc9204671cda811d2b488ce983c540e074a4d994db8f` |
| FSC mapping table `configs/fsc_mapping_v1.json` | `7a385e29c67bf9ac57ac034aa4149583b0d5432179cea71e9e1363d48f2f0ced` |
| FSC report `outputs/fsc_ext_v1/score/report.json` | `dc694398b0ff00f22b4e7eccc6aa655b4e1c63da65fe089e21521299193218e5` |

**SLURP split manifests** (split_v2_seed2026):
- The runtime path is `data/slurp/manifests/split_v2_seed2026/`. `data` is a symbolic link to the shared storage `<SHARED_ROOT>/data`, and git cannot track files behind a link, so the repository stores **byte-identical copies** in `manifests/slurp_split_v2_seed2026/` (the original files of 2026-11-05, including the original `.csv.gz`).
- To restore: `cp manifests/slurp_split_v2_seed2026/* data/slurp/manifests/split_v2_seed2026/`.

| File | sha256 (file itself) | sha256 (gzip-decompressed content) |
|---|---|---|
| `train_recordings.csv` | `782f458940665b5010f2be5b0eaa497fe3ee52c5a3d7cbe2bd4df12368f125f9` | — |
| `devel_recordings.csv` | `08ebe28425aa311b89d1376d3e24cd1b15c3f6fe30489e1afa36aab6ebc21498` | — |
| `test_recordings.csv` | `81efdf51e7c6017b5840eb73a53158e3197b7eb9782c3a70a365b7629242df7b` | — |
| `train_candidate_pairs.csv.gz` | `d2515a16f4ca955a19177bdc0237e9399aa0f1c8a9f14be9f5a34ec3bd819f77` | `d3edcf7add7cacfc75748d5e8d1ecab7c10c58b19b8d01089df622c82bf1a90d` |
| `devel_candidate_pairs.csv.gz` | `6bf58c549b368cd6ecdc8aaf9d49014af8b9003a95cfe423d48e406ee8fe132f` | `cc2ca3f4c6ca4f0fa4a7c0c2eb531a699331464482979903c1ec3507f31bef37` |
| `test_candidate_pairs.csv.gz` | `383a32e69c04a49a47d62362e4606d42ad2a58554b249d1b764810e5b8f8d375` | `fd16f01c0d088fa2b2054e9d0f0e1e3cb48277f06c115e1c455fe8e615a01b50` |
| Input: `results/slurp/headset_durations.csv` (recording-duration table, input for rebuilding the split, tracked) | `0af9a40364573daa8651aaafc9351a4bff8f62daf4e43a6ab1dfd8eb955ee04e` | — |

**Rebuilding the split**: `python -I scripts/slurp/make_slurp_split_v2.py data/slurp/annotations/8eb1654 results/slurp/headset_durations.csv data/slurp/manifests/split_v2_seed2026` (**relative paths recommended**; 2 CPUs, 8 GB, about 40 minutes). The split was rebuilt and checked on CPU on 2026-11-13 (record at the end of `docs/slurp_test_freeze_v1.md`):
- the three `*_recordings.csv` files are byte-identical;
- the **file** sha256 of the three `.csv.gz` files differs because of the write timestamp in the gzip header; **the sha256 of the decompressed content is authoritative** (the rebuilt versions are byte-identical after decompression);
- the script also rewrites `configs/slurp_split_v2_seed2026.json` and `results/slurp/split_v2_stats.json`. The differences are only in path fields (when absolute paths are passed) and in the last floating-point digits, and **do not affect the manifest content**.

`data/`, `outputs/`, checkpoints and audio are not in the repository (see `.gitignore`); all of these paths are on shared storage.

## Data licences
- **SLURP audio** is CC BY-NC 4.0, non-commercial use only. Derived audio generated from SLURP in this project (reverberant, mixed, separated outputs) is **not published**.
- **FSC**: please obtain FSC from the official Fluent.ai channel (the official dataset page and licence are listed in `configs/fsc_pilot_pending.json`). This project uses FSC only for **non-commercial academic research**; redistribution authorization for the copy used **has not yet been confirmed by Fluent.ai** and must be confirmed before publication. Acquisition and audit: see `docs/fsc_pilot_plan.md`. Derived audio is likewise **not published**; the FSC official test set was not used.
- **SMS-WSJ RIRs**: used locally only and not redistributed.


## Path placeholders
In the public version, cluster-specific paths and settings are replaced by the placeholders below (in code, configs and documents). Replace them with your own values before running.

| Placeholder | Meaning | Replace with |
|---|---|---|
| `<REPO_ROOT>` | Absolute path of this repository | The directory where you cloned this repository |
| `<SHARED_ROOT>` | Project data root: the shared-storage directory holding data (`data/`), environments (`envs/`) and checkpoints (`checkpoints/`, including the HF cache `checkpoints/hf`) | The directory where you keep SLURP audio, generated data, environments and weights |
| `<SEPARATION_DATA_ROOT>` | SMS-WSJ data root (containing `data/sms_wsj/sms_wsj.json` and `rirs/`) | Your SMS-WSJ generation directory |
| `<SEPARATION_REPO>` | Separation front-end code repository (SpatialNet / TF-GridNet training code and experiment directories) | Your separation front-end code directory |
| `<SHARED_STORAGE>` | Parent directory of the shared storage | Same as above, following your storage layout |
| `<CONDA_ROOT>` | Personal conda installation directory (where the `tfgridnet` environment lives) | Your conda directory |
| `<HOME>` | User home directory | Your home directory |
| `<USER>` | Cluster user name | Your user name |
| `<SLURM_ACCOUNT>` | Slurm account (`-A` / `--account`) | Your Slurm account |
| `<GPU_PARTITION>` / `<CPU_PARTITION>` / `<PARTITION>` | Slurm GPU / CPU partitions | The partition names on your cluster |
| `<NODE>` | Compute-node names in run records (documents only) | No replacement needed |
| `<GROUP>`, `<PROJECT_GROUP>`, `<CLUSTER_DOMAIN>`, `<EMAIL>` | Cluster group name, project group name, cluster domain, e-mail (documents only) | No replacement needed |
| `<FSC utterance>` | One FSC utterance removed from a document | No replacement needed |

## Licence and citation
- Author: Yunqi Li. Repository: https://github.com/yunqiyunqili/audio-llm-target-speaker
- Code and documents: MIT (`LICENSE`, Copyright (c) 2026 Yunqi Li). Data and model licences: `DATA_LICENSES.md`; citation: `CITATION.cff`.
- Note: the sha256 values of `configs/slurp_split_v2_seed2026.json`, `configs/mix_v1_test_v1.json` and `configs/slurp_test_v1_s1_sha256.txt` recorded in the report and the freeze document are those of the original run versions; in the public version these three files differ only in path fields, and their sha256 values are listed in the "Key file sha256" table above.
- Cluster paths, accounts, partitions and node names in the public version have been replaced by placeholders such as `<SHARED_ROOT>`, `<REPO_ROOT>`, `<SLURM_ACCOUNT>`, `<GPU_PARTITION>`, `<NODE>`. Replace them with your own paths and cluster settings before running the scripts.
- The per-utterance FSC sample list (`configs/fsc_ext_v1_list.json`) is not included in the public version. It can be rebuilt from your own copy of FSC following the rules in `scripts/fsc/ext_build_list.py`.
