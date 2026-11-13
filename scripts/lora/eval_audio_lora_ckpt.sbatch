#!/bin/bash
# Step-1 audio LoRA checkpoint inference (docs/lora_v2_runs.md §15, §17). ONE checkpoint per job, 20 min.
# Usage: sbatch scripts/lora/eval_audio_lora_ckpt.sbatch <lr> <step> <items devel_clean|devel_rev> <half selection|report|all>
#SBATCH --job-name=lora-audio-eval
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_audio_eval_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/lora/eval_audio_lora_ckpt.py --run-dir $SHARED/checkpoints/lora_v2/step1/lr$1 --step $2 --items $3 --half $4 \
  --out "outputs/lora_audio_eval/lr$1/${SLURM_JOB_ID}"
