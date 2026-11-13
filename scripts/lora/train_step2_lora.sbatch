#!/bin/bash
# Step-2 Whisper + T2 LoRA training segment (docs/lora_slu_plan_v2.md; docs/lora_v2_runs.md §15). TRAIN split only. 30 min per segment.
# Usage: sbatch scripts/lora/train_step2_lora.sbatch <lr> <segment K>            (training; run dir step2/lr<lr>)
#        sbatch scripts/lora/train_step2_lora.sbatch <lr> 1 pilot <N>            (pilot; run dir step2_pilot/<jobid>)
#SBATCH --job-name=lora-step2-train
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_step2_train_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ "${3:-}" = pilot ]; then
  python scripts/lora/train_step2_lora.py --lr $1 --segment 1 --pilot $4 --run-dir $SHARED/checkpoints/lora_v2/step2_pilot/${SLURM_JOB_ID}
else
  python scripts/lora/train_step2_lora.py --lr $1 --segment $2 --run-dir $SHARED/checkpoints/lora_v2/step2/lr$1
fi
