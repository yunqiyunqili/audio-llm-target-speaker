#!/bin/bash
# Snapshot front-end downstream: selected stream -> P2 and Whisper + T2 (docs/frontend_compare_v1.md). IN-TRAINING snapshot; diagnosis only.
# Usage: sbatch scripts/qwen/qwen_omni_snapshot.sbatch <model> <snapshot_dir_name>
#SBATCH --job-name=qwen-omni-snap
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/qwen_omni_snapshot_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
M=$1; SNAP=$2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/qwen/qwen_omni_snapshot.py --model $M --sep-dir $SHARED/data/mix_v1/separated_v1_snapshot/$SNAP \
  --selection outputs/frontend_snapshot_eval/$M/selection.jsonl --out "outputs/qwen_omni_snapshot/${SLURM_JOB_ID}_$M"
