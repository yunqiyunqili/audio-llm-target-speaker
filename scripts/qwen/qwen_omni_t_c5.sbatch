#!/bin/bash
# Qwen2.5-Omni-7B T_C5 rerun only (existing Whisper W_C5 transcripts -> T2); docs/c7_whisper_v1.md.
# Config: 1x A100, 4 CPU, 48 GB, 30 min (same as qwen_omni_single).
#SBATCH --job-name=qwen-omni-tc5
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/qwen_omni_t_c5_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__)"
python scripts/qwen/qwen_omni_t_c5.py --w-c5 outputs/qwen_omni_c7w_f1/22016221/W_C5/raw.jsonl --out "outputs/qwen_omni_c7w_f1_tc5/${SLURM_JOB_ID}"
