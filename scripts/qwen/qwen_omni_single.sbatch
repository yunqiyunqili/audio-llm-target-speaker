#!/bin/bash
# Qwen2.5-Omni-7B text-only single-utterance inference (C1 original + C1b 8k->16k).
# Approved 2026-11-03: env envs/qwen_omni (torch 2.6.0+cu124, transformers 4.52.4), weights ae9e1690,
# SLURP audio extracted. Config: 1x A100, 4 CPU, 48 GB, 30 min.
# Resubmission (approved once): model loaded from local snapshot dir of commit ae9e1690; offline kept.
#SBATCH --job-name=qwen-omni-single
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/qwen_omni_single_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
AUDIO=$SHARED/data/slurp/audio/slurp_real/audio--1504197928-headset.flac   # test slurp_id 0, rule A
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__)"
python scripts/qwen/qwen_omni_single.py --audio "$AUDIO" --out "outputs/qwen_omni_single/${SLURM_JOB_ID}"
