#!/bin/bash
#SBATCH --job-name=fsc-spatialnet-pilot
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=<REPO_ROOT>/outputs/logs/fsc_spatialnet_%j.out
set -euo pipefail
cd <REPO_ROOT>
export PATH=<CONDA_ROOT>/envs/tfgridnet/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python scripts/fsc/spatialnet_infer.py --inputs outputs/fsc_pilot/spatial_audio_real --out outputs/fsc_pilot/spatialnet_ep100
