#!/bin/bash
# mix_spec_v1 audio generation: 100 groups x 2 swaps from configs/mix_v1_devel100_v2.json (docs/mix_spec_v1.md).
# CPU only. Writes ~332 MB float32 wav to the shared data dir.
#SBATCH --job-name=mix-v1-devel
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/mix_slurp_v1_%j.out
set -euo pipefail
cd <REPO_ROOT>
SHARED=<SHARED_ROOT>
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
envs/qwen_omni/bin/python -I scripts/sms_wsj/mix_slurp_v1.py configs/mix_v1_devel100_v2.json \
  $SHARED/data/slurp/audio/slurp_real $SHARED/data/mix_v1/devel_seed2026
