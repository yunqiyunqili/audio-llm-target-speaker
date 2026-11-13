#!/bin/bash
# SI-SDR vs direct / early / reverberant mic-0 references for BLSTM, oracle and cACGMM MVDR.
# NOT SUBMITTED - awaiting approval. Submit from project root: sbatch scripts/sms_wsj/rev_ref_sisdr.sbatch
#SBATCH --job-name=rev-ref-sisdr
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/rev_ref_sisdr_%j.out
set -euo pipefail
cd <REPO_ROOT>
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
python -B scripts/sms_wsj/rev_ref_sisdr.py --out "outputs/rev_ref_sisdr/${SLURM_JOB_ID}"
