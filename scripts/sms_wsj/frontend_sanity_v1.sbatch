#!/bin/bash
# Front-end sanity checks A/B/C (docs/frontend_sanity_v1.md). Metrics JSON only; no audio written.
#SBATCH --job-name=fe-sanity-v1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/frontend_sanity_v1_%j.out
set -euo pipefail
cd <REPO_ROOT>
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
echo "GIT_COMMIT $(git rev-parse HEAD) DIRTY $(git status --porcelain | wc -l)"
for m in blstm_mvdr fasnet_tac; do
  python -B scripts/sms_wsj/frontend_sanity_v1.py --model $m --out "outputs/frontend_sanity_v1/${SLURM_JOB_ID}"
done
