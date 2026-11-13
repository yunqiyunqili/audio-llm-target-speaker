#!/bin/bash
# Zero-shot BLSTM+MVDR and FaSNet-TAC on the 200 mix_v1 observations (docs/frontend_mix_v1_plan.md).
#SBATCH --job-name=fe-mix-v1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/frontend_mix_v1_%j.out
set -euo pipefail
cd <REPO_ROOT>
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
M=<SHARED_ROOT>/data/mix_v1
echo "GIT_COMMIT $(git rev-parse HEAD) DIRTY $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for m in blstm_mvdr fasnet_tac; do
  python -B scripts/sms_wsj/frontend_mix_v1.py --model $m --mix-dir $M/devel_seed2026 --out-root $M/separated_v1
done
