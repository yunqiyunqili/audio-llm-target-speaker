English translation of repro_plan.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：repro_plan.md

# Reproducing the SMS-WSJ test metrics of BLSTM+MVDR and FaSNet-TAC (plan, not yet submitted)

## Goal
Rerun inference with the existing `best.pt` on `test_eval92` (1,332 utterances) and confirm that the recorded values are obtained:

| Model | Recorded SI-SDR | Recorded SI-SDRi | Recorded mixture SI-SDR | Source |
|---|---|---|---|---|
| BLSTM+MVDR | −1.0427572 | 4.4121033 | −5.4548605 | `MC/blstm_mvdr/exp/sms_blstm_mvdr_v1/test_summary.json` |
| FaSNet-TAC | 9.4404093 | 14.8952699 | −5.4548605 | `MC/fasnet_tac/exp/sms_fasnet_tac_v1/test_summary.json` |

This is an **evaluation reproduction** (same weights, same code, same data), with no retraining.

## Method
- `scripts/sms_wsj/repro_sms_wsj_test.py` imports the original `<model>/train.py` directly and calls its own `evaluate()`, without rewriting any evaluation logic.
- The original `evaluate()` writes results into run_dir and hashes `run_dir/best.pt`. The script therefore points run_dir at `outputs/repro_sms_wsj/<model>_<jobid>/`, a directory containing only a symlink to the original `best.pt`. To avoid writing into `../`, `-B` and `PYTHONDONTWRITEBYTECODE` are set. The paths and time ranges actually checked are in "actual check of whether anything was written to `../`" of `docs/log/day1_report.md`.
- Checks before inference (repeated inside the GPU job):
  1. the sha256 of `best.pt` matches the record;
  2. the config in the checkpoint matches `run_config.json`; the selected epoch matches;
  3. recompute with the original `configuration()` the hashes, audit and manifest hash of the current code and the ESPnet/TAC sources, and check that they match the values stored in the checkpoint (confirming that code and data have not changed since training).
- After inference: compare the means, and compare item by item the SI-SDR, SI-SDRi, mixture SI-SDR and permutation in the original `test_utterances.jsonl`; results are written to `comparison.json`.
- Verdict: mean difference ≤ 1e-3 dB, maximum per-item difference ≤ 1e-2 dB, and all permutations identical → match, exit code 0; otherwise exit code 3. The actual differences are recorded whether or not they match. The original run used deterministic cuDNN with TF32 disabled, so near bit-exact results are expected; but if the assigned node has a different driver, tiny differences may appear; measurements take precedence.

## Verification already done on the login node (CPU checks only, no inference)
`python -B scripts/sms_wsj/repro_sms_wsj_test.py --model {blstm_mvdr,fasnet_tac} --check-only`: all 5 provenance checks are true for both models, test set 1,332 utterances → `CHECK_ONLY_PASS`.

## Job configuration (not submitted)
| | BLSTM+MVDR | FaSNet-TAC |
|---|---|---|
| Script | `scripts/sms_wsj/repro_blstm_mvdr_test.sbatch` | `scripts/sms_wsj/repro_fasnet_tac_test.sbatch` |
| account / partition | <SLURM_ACCOUNT> / <GPU_PARTITION> (same as the original training) | same as left |
| Resources | 1× A100, 4 CPU, 32 GB RAM, limit 1 h | same as left |
| Environment | `<CONDA_ROOT>/envs/tfgridnet` (same as the original run, torch 2.5.1+cu121) | same as left |
| Original test inference time | 73 s | 91 s |
| Original peak GPU memory | 0.43 GiB | 4.68 GiB |
| Output | `outputs/repro_sms_wsj/blstm_mvdr_<jobid>/` | `outputs/repro_sms_wsj/fasnet_tac_<jobid>/` |
| Log | `outputs/logs/repro_blstm_mvdr_<jobid>.out` | `outputs/logs/repro_fasnet_tac_<jobid>.out` |
| New disk usage | about 0.3 MB (per-item jsonl + summary + symlink) | same as left |

Submission commands (to be executed only after you confirm, run from the project root):
```
sbatch scripts/sms_wsj/repro_blstm_mvdr_test.sbatch
sbatch scripts/sms_wsj/repro_fasnet_tac_test.sbatch
```
