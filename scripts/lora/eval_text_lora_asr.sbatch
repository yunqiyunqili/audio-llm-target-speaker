#!/bin/bash
# DIAGNOSTIC: step-0 text LoRA on Whisper transcripts of devel clean 428 (docs/lora_v2_runs.md §13-3). <= 20 min per part.
# Usage: sbatch scripts/lora/eval_text_lora_asr.sbatch <lr> <step> <part> <nparts> <W raw.jsonl> [<W raw.jsonl> ...]
#SBATCH --job-name=lora-text-asr
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/lora_text_asr_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export PATH=$SHARED/envs/qwen_omni_lora/bin:$PATH
export HF_HOME=$SHARED/checkpoints/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
LR=$1 STEP=$2 PART=$3 NP=$4; shift 4
python scripts/lora/eval_text_lora_asr.py --run-dir $SHARED/checkpoints/lora_v2/step0/lr$LR --step $STEP --part $PART --nparts $NP --asr "$@" \
  --out "outputs/lora_text_asr/lr${LR}_step${STEP}/part${PART}of${NP}_${SLURM_JOB_ID}"
