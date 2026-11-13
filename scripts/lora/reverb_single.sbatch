#!/bin/bash
# Single-talker full-reverb audio for LoRA step 1 (docs/lora_slu_plan_v2.md §3.1; docs/lora_v2_runs.md §10). CPU, one part per job, <= 20 min.
# Usage: sbatch scripts/lora/reverb_single.sbatch <split devel|train|test> <part> <nparts>
#SBATCH --job-name=lora-reverb
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_reverb_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
case $1 in devel) OUT=$SHARED/data/lora_v1/devel_rev_cv_dev93;; train) OUT=$SHARED/data/lora_v1/train_rev_si284;; test) OUT=$SHARED/data/lora_v1/test_rev_eval92;; esac
envs/qwen_omni/bin/python -I scripts/lora/reverb_single.py --split $1 --part $2 --nparts $3 --out $OUT
