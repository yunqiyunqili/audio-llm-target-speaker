#!/bin/bash
# Qwen2.5-Omni-7B C1 structured SLU pilot on 20 devel recordings (plan: docs/c1_semantic_plan.md).
# NOT SUBMITTED. Config: 1x A100, 4 CPU, 48 GB, 30 min (same as qwen_omni_single).
#SBATCH --job-name=qwen-omni-c1sem
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/qwen_omni_c1_semantic_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__)"
python scripts/qwen/qwen_omni_c1_semantic.py --audio-dir $SHARED/data/slurp/audio/slurp_real --out "outputs/qwen_omni_c1_semantic/${SLURM_JOB_ID}"
