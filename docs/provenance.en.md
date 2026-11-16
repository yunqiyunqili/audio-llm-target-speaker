English translation of provenance.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：provenance.md

# Data and environment provenance record (permanently valid)

## SLURP version
- All subsequent SLURP processing uses the annotations of commit **8eb16545762be97ace75334109d73824217311f1** (short 8eb1654) of GitHub pswietojanski/slurp, at `data/slurp/annotations/8eb1654/`; the audio is slurp_real from Zenodo 4274930 (md5 checked).
- The decision to deviate from the official split and the reasons are in `docs/log/day1_followup4.md` §6.
- Differences between the paper and the released data **have not been resolved** (checked 2026-11-03, see `docs/log/day1_followup3.md` §7):
  - close-talk recordings: the released data have 34,663 unique files, the paper's Table 2 has 34,603, a difference of 60. The released annotations and audio are consistent with each other (0 missing, 0 extra).
  - speakers: 174 usrids in metadata.json, 177 in the paper's Table 2, a difference of 3.
  - In addition, 1 file, `audio-1502309360-headset.flac`, is attached to both slurp_id 15959 and 16142 (with different labels), so summing over splits counts 1 extra (34,664); all derived lists exclude this file.
- When reporting SLURP counts, the released data of commit 8eb1654 take precedence, with a note that they differ from the paper's numbers.

## conda writes in the home directory (produced when creating envs/qwen_omni, 2026-11-03 17:18)
`conda create -p <SHARED_ROOT>/envs/qwen_omni` wrote 3 files in the home directory, all automatic conda behaviour, not edited manually:

| File | Size | Content |
|---|---|---|
| `~/.conda/environments.txt` | 306 B | one line appended, registering the new environment path |
| `~/.conda/aau_token` | 22 B | conda anonymous usage identifier, updated |
| `~/.conda/aau_token_host` | 8 B | same as above |

Apart from this, no existing environment was changed (`<CONDA_ROOT>/envs/*` and `speech_separation/envs/*` have no files modified after 17:00).

## Qwen2.5-Omni-7B weights
- `Qwen/Qwen2.5-Omni-7B` commit ae9e1690543ffd5c0221dc27f79834d0294cba00, local snapshot directory
  `<SHARED_ROOT>/checkpoints/hf/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00`.
- The cache has no `refs/main`; the inference scripts load directly from this directory (offline), and the commit is fixed by the directory.

## A job not submitted by this project: 22009884 (sms-grid-6ch, TF-GridNet 6ch)
- Conclusion: this job was not submitted by this project; it appears to have been resubmitted automatically by the original training script when the previous segment ended. This project did not touch it, nor any TF-GridNet / SpatialNet job.
- Basis (read-only queries, 2026-11-04):
  1. This project's submission record `outputs/logs/submitted_jobs.txt` does not contain 22009884.
  2. `sacct`: the previous segment job 21989292 ended at 2026-11-04T21:26:43 with status COMPLETED; the submission time of 22009884 is also exactly 2026-11-04T21:26:43. The WorkDir of both jobs is `../speech_separation/multichannel`, and the SubmitLine is the same, `sbatch --parsable --job-name=sms-grid-6ch train.sbatch 6 tfgridnet/exp/sms_direct_6ch_v1`.
  3. Lines 20–23 of `../speech_separation/multichannel/train.sbatch`: when `train.py` exits with code 75, the script runs `sbatch --parsable --job-name="sms-grid-${channels}ch" train.sbatch ...`, appends the new job ID to `$run_dir/continuation_jobs.txt`, and prints `CONTINUATION_SUBMITTED`.
  4. The end of `../speech_separation/multichannel/logs/train_21989292.out` is, in order, epoch 14 (dev SI-SDR 17.19 dB), `CONTINUATION_REQUIRED`, `CONTINUATION_SUBMITTED 22009884`.

## SLURP split version
- **From now on, `data/slurp/manifests/split_v2_seed2026/` is used throughout**; configuration in `configs/slurp_split_v2_seed2026.json`, description in `docs/log/day1_followup4.md` §9. Rules: intent is scenario_action, basis A, seed 2026, test 40%; devel is 20% of the remaining speakers of known gender, and its sentences are all official devel sentences plus 35% of the official train sentences.
- `data/slurp/manifests/heldout_seed2026/` (A_*, S_*, S_dedup_*) is **deprecated** (2026-11-04); a `DEPRECATED.md` has been added to the directory, and no files were deleted. `configs/slurp_heldout_speakers_seed2026.json` is likewise kept for traceability only.
- The sensitivity list (requiring only status=correct), if needed later, will be regenerated under the v2 rules.
- Accepted limitations (user decision 2026-11-04): v2 devel has only 11 speakers with recordings and 428 recordings; with each recording used at most 3 times, same gender and duration difference ≤1 s, there are 197 pairs. The total-variation distance of entity types between devel and train is 0.264, larger than the 0.201 between test and train.
- Whether semantically similar entity types (e.g. time / timeofday) are mixed up is not checked for now; if such problems appear among the C1 misjudged samples, they will be reported.
