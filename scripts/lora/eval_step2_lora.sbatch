#!/bin/bash
# Step 2: one checkpoint on frozen Whisper transcripts; <=20min per job.
# Usage: sbatch scripts/lora/eval_step2_lora.sbatch LR STEP ITEMS HALF W_RAW [W_RAW ...]
#SBATCH --job-name=lora-step2-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_step2_eval_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
LR=$1 STEP=$2 ITEMS=$3 HALF=$4; shift 4
python scripts/lora/eval_step2_lora.py --run-dir $SHARED/checkpoints/lora_v2/step2/lr$LR --step $STEP --items $ITEMS --half $HALF --asr "$@" \
  --out "outputs/lora_step2_eval/lr${LR}/step${STEP}/${ITEMS}_${HALF}_${SLURM_JOB_ID}"
