#!/bin/bash
# Re-evaluate the finished blstm_mvdr SMS-WSJ checkpoint on test_eval92 (1,332 utt) and
# compare with the recorded test metrics. Inference only; no training.
# NOT SUBMITTED. Submit from the project root:  sbatch scripts/sms_wsj/repro_blstm_mvdr_test.sbatch
#SBATCH --job-name=repro-blstm-mvdr
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/repro_blstm_mvdr_%j.out
set -euo pipefail
cd <REPO_ROOT>
# Same interpreter and determinism settings as the original training/eval job.
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1   # do not write __pycache__ into ../speech_separation
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -B scripts/sms_wsj/repro_sms_wsj_test.py --model blstm_mvdr \
    --out "outputs/repro_sms_wsj/blstm_mvdr_${SLURM_JOB_ID}" --workers 4
