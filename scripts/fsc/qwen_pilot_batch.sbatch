#!/bin/bash
# FSC pilot formal run (docs/fsc_pilot_plan.md, 2026-11-11): one route x 256 tasks per job.
# Usage: sbatch scripts/fsc/qwen_pilot_batch.sbatch <base|audio_lora|t2_lora>
#SBATCH --job-name=fsc-pilot-v2
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/fsc_pilot_v2_%x_%j.out
set -euo pipefail
cd <REPO_ROOT>
ROUTE=$1
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
test -z "$(git status --porcelain)" || { echo DIRTY_WORKTREE; git status --porcelain; exit 3; }
echo GIT_COMMIT $(git rev-parse HEAD); echo PYTHON $(which python); echo ROUTE $ROUTE
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
TASKS=outputs/fsc_pilot/pilot_tasks.jsonl
echo "9f9cc65db6ca3b81f1f208a8ee4b9a1d258fcbd265fa9ed923d35af850fd9e36  $TASKS" | sha256sum -c -
python scripts/fsc/qwen_pilot_batch.py --route $ROUTE --tasks $TASKS --out outputs/fsc_pilot/pred_v2_${ROUTE}_${SLURM_JOB_ID}
