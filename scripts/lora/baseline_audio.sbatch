#!/bin/bash
# Audio baselines / transcription for LoRA steps 1-2 (docs/lora_slu_plan_v2.md; docs/lora_v2_runs.md §10). One part per job, <= 20 min.
# Usage: sbatch scripts/lora/baseline_audio.sbatch <mode p2|whisper_t2|whisper> <items devel_clean|devel_rev|train_all> <part> <nparts>
#SBATCH --job-name=lora-base-audio
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_base_audio_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/lora/baseline_audio.py --mode $1 --items $2 --part $3 --nparts $4 --out "outputs/lora_base_audio/$1_$2/part$3of$4_${SLURM_JOB_ID}"
