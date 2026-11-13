#!/bin/bash
# Approved 30-minute exception for final base-model downstream, one route per job.
# Usage: sbatch qwen_omni_final.sbatch SEPARATION_DIR SELECTION_JSON ROUTE_LABEL
#SBATCH --job-name=qwen-final
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/qwen_final_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/qwen/qwen_omni_snapshot.py --model spatialnet --final --sep-dir "$1" --selection "$2" --out "outputs/qwen_omni_final/${SLURM_JOB_ID}_$3"
