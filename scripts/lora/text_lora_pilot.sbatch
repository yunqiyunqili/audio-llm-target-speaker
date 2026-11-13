#!/bin/bash
# Step-0 text LoRA PILOT (docs/lora_slu_plan_v2.md, frozen): speed + memory only, train split only. 30 min.
# Usage: sbatch scripts/lora/text_lora_pilot.sbatch
#SBATCH --job-name=lora-text-pilot
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_text_pilot_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/lora/text_lora_pilot.py --out "outputs/lora_text_pilot/${SLURM_JOB_ID}"
