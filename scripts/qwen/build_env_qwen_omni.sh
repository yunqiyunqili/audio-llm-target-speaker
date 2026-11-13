#!/bin/bash
# Build the Qwen2.5-Omni inference env (approved 2026-11-03). Run on a CPU node via srun:
#   srun -A <SLURM_ACCOUNT> -p <CPU_PARTITION> -c 4 --mem=16G -t 01:00:00 bash scripts/qwen/build_env_qwen_omni.sh
# Creates a NEW prefix only; existing envs are not touched. Caches live on the shared disk.
set -euo pipefail
S=<SHARED_ROOT>
P=$S/envs/qwen_omni
PROJ=<REPO_ROOT>
export CONDA_PKGS_DIRS=$S/envs/.conda_pkgs PIP_CACHE_DIR=$S/envs/.pip_cache XDG_CACHE_HOME=$S/envs/.cache
export PYTHONNOUSERSITE=1
test ! -e "$P" || { echo "ERROR: $P already exists"; exit 1; }

<CONDA_ROOT>/bin/conda create -y -p "$P" python=3.10 pip
PY="$P/bin/python"
"$PY" -m pip install --no-input --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchaudio==2.6.0
"$PY" -m pip install --no-input \
    transformers==4.52.4 accelerate==1.7.0 qwen-omni-utils==0.0.8 \
    soundfile==0.13.1 scipy==1.15.3 librosa==0.11.0
"$PY" -m pip check
"$PY" - <<'EOF'
import torch, torchaudio, transformers, accelerate
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # import check only
from transformers.utils.import_utils import is_torch_greater_or_equal
assert torch.__version__.startswith('2.6.0') and torch.version.cuda == '12.4', (torch.__version__, torch.version.cuda)
assert transformers.__version__ == '4.52.4', transformers.__version__
print('IMPORT_OK', torch.__version__, torch.version.cuda, torchaudio.__version__, transformers.__version__,
      accelerate.__version__, 'torch>=2.6:', is_torch_greater_or_equal('2.6'))
EOF
{ echo "# envs/qwen_omni ($P) frozen $(date -Is) on $(hostname)"
  echo "# python $("$PY" -c 'import sys;print(sys.version.split()[0])'); torch from https://download.pytorch.org/whl/cu124"
  "$PY" -m pip freeze; } > "$PROJ/docs/env_qwen_omni.txt"
du -sh "$P" "$S/envs/.conda_pkgs" "$S/envs/.pip_cache" 2>/dev/null || true
echo BUILD_DONE
