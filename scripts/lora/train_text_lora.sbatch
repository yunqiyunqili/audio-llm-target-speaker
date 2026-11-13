#!/bin/bash
# Step-0 text LoRA training segment (docs/lora_slu_plan_v2.md + revision 1). TRAIN split only. 30 min per segment.
# Usage: sbatch scripts/lora/train_text_lora.sbatch <lr> <segment 1|2>
#SBATCH --job-name=lora-text-train
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_text_train_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/lora/train_text_lora.py --lr $1 --segment $2 --run-dir $SHARED/checkpoints/lora_v2/step0/lr$1
