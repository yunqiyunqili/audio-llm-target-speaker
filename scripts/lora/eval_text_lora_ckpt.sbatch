#!/bin/bash
# Step-0 checkpoint inference on devel reference transcripts (docs/lora_slu_plan_v2.md + revision 1). 30 min.
# Usage: sbatch scripts/lora/eval_text_lora_ckpt.sbatch <lr> <steps comma list> <half>
#SBATCH --job-name=lora-text-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_text_eval_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/lora/eval_text_lora_ckpt.py --run-dir $SHARED/checkpoints/lora_v2/step0/lr$1 --steps $2 --half $3 \
  --out "outputs/lora_text_eval/lr$1/${SLURM_JOB_ID}"
