#!/bin/bash
# Step-1 audio LoRA on mix_v1 C4-direct / C3 (docs/lora_v2_runs.md §21). ONE condition per job, 20 min.
# Usage: sbatch scripts/lora/eval_audio_lora_mix.sbatch <lr> <step> <cond c4direct|c3>
#SBATCH --job-name=lora-audio-mix
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_audio_mix_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/lora/eval_audio_lora_mix.py --run-dir $SHARED/checkpoints/lora_v2/step1/lr$1 --step $2 --cond $3 --out "outputs/lora_audio_mix/lr$1/${SLURM_JOB_ID}"
