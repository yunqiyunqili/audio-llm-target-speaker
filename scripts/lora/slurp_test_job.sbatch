#!/bin/bash
# SLURP test job wrapper (docs/slurp_test_freeze_v1.md §7). Resources are given on the sbatch command line:
#   GPU: -p <GPU_PARTITION> --gres=gpu:a100:1 -c 4 --mem=48G -t 00:20:00 (SpatialNet: --mem=32G)
#   CPU: -p <CPU_PARTITION> -c 4 --mem=16G -t 00:20:00
# Usage: sbatch -A <SLURM_ACCOUNT> -J <name> <resources> scripts/lora/slurp_test_job.sbatch <env qwen_omni|qwen_omni_lora|tfgridnet|sms_wsj> <python args...>
# Sets SLURP_EVAL_SPLIT=test, refuses to run on a dirty worktree, records commit / node / GPU. No automatic retry.
#SBATCH --output=<REPO_ROOT>/outputs/logs/slurp_test_%x_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
ENV=$1; shift
case $ENV in
  qwen_omni|qwen_omni_lora) export PATH=$SHARED/envs/$ENV/bin:$PATH ;;
  tfgridnet) export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH; export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 ;;
  sms_wsj) export PATH=<SEPARATION_REPO>/envs/sms_wsj/bin:$PATH ;;
  *) echo "UNKNOWN_ENV $ENV"; exit 2 ;;
esac
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 SLURP_EVAL_SPLIT=${EVAL_SPLIT:-test}  # EVAL_SPLIT=fsc_ext for the FSC extension
test -z "$(git status --porcelain)" || { echo DIRTY_WORKTREE; git status --porcelain; exit 3; }
echo "GIT_COMMIT $(git rev-parse HEAD) NODE ${SLURMD_NODENAME:-} ENV $ENV PYTHON $(which python) SPLIT $SLURP_EVAL_SPLIT"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
python "$@"
echo JOB_DONE
