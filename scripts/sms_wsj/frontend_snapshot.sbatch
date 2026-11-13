#!/bin/bash
# Snapshot front-end diagnosis (docs/frontend_compare_v1.md): cv_dev93 check, then mix_v1 inference if PASS. IN-TRAINING snapshots.
#SBATCH --job-name=fe-snapshot
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/frontend_snapshot_%j.out
set -uo pipefail
cd <REPO_ROOT>
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
S=<SHARED_ROOT>
echo "GIT_COMMIT $(git rev-parse HEAD) DIRTY $(git status --porcelain | wc -l) NODE $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for spec in tfgridnet:tfgridnet_ep16_20261105-211918 spatialnet:spatialnet_ep72_20261105-211918; do
  m=${spec%%:*}; snap=${spec#*:}
  python -B scripts/sms_wsj/frontend_snapshot.py --model $m --snapshot $S/checkpoints/snapshots/$snap --mode cv --out "outputs/frontend_snapshot/${SLURM_JOB_ID}/$m" \
  && python -B scripts/sms_wsj/frontend_snapshot.py --model $m --snapshot $S/checkpoints/snapshots/$snap --mode mix --out "outputs/frontend_snapshot/${SLURM_JOB_ID}/$m" \
       --sep-out $S/data/mix_v1/separated_v1_snapshot/$snap \
  || echo "STOPPED $m (see cv_check.json / traceback)"
done
