#!/bin/bash
# Approved final SpatialNet cv_dev93 check followed by mix_v1 inference.
# Usage: sbatch frontend_final.sbatch FINAL_CHECKPOINT_DIR NEW_SEPARATION_DIR
#SBATCH --job-name=fe-spatial-final
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/frontend_final_%j.out
set -euo pipefail
cd <REPO_ROOT>
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
echo "GIT_COMMIT $(git rev-parse HEAD) DIRTY $(git status --porcelain | wc -l) NODE $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -B scripts/sms_wsj/frontend_snapshot.py --model spatialnet --snapshot "$1" --final --mode cv --out "outputs/frontend_final/${SLURM_JOB_ID}/spatialnet"
python -B scripts/sms_wsj/frontend_snapshot.py --model spatialnet --snapshot "$1" --final --mode mix --out "outputs/frontend_final/${SLURM_JOB_ID}/spatialnet" --sep-out "$2"
