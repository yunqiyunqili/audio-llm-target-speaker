#!/bin/bash
# C0 devel-428 baseline: reference transcripts of all 428 split_v2 devel recordings -> frozen T2 (docs/lora_slu_plan_v2.md).
# Usage: sbatch scripts/qwen/qwen_omni_c0_devel428.sbatch
#SBATCH --job-name=qwen-omni-c0-428
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/qwen_omni_c0_devel428_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/qwen/qwen_omni_c0_devel428.py --out "outputs/qwen_omni_c0_devel428/${SLURM_JOB_ID}"
