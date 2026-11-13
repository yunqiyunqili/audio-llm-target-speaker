"""FSC pilot audio preprocessing, shared by inference (qwen_pilot_batch.py) and the CPU assertion check.

Each condition maps to exactly one SLURP chain (docs/fsc_pilot_plan.md, 2026-11-11 fix):
  clean                  -> SLURP C1   : original 16 kHz mono, no level change, no chain8.
  reverb                 -> SLURP C1-R : file produced by scripts/lora/reverb_single.reverb() (16k -> 8k, cv_dev93 RIR mic 0,
                                         rir_start alignment, 8k conv -> resample_poly(2,1) -> level() -30 dBFS), used as is.
  spatial / direction_uncertainty -> SLURP C5 : 8 kHz SpatialNet out{k}.wav -> polarity sign (once) -> chain8 (once)
                                         = resample_poly(x,2,1)[:2n] -> level().
The same waveform goes to Qwen (P2 routes) and to Whisper (T2 route). Nothing else is applied afterwards.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/qwen'))
from qwen_omni_c7w_f1 import chain8  # noqa: E402

SPATIAL = ('spatial', 'direction_uncertainty')


def prepare(it):
    """Return (waveform float32 @16 kHz, info dict). info['n_chain8'] counts chain8 applications."""
    if it['condition'] in SPATIAL:
        x, sr = sf.read(it['audio_path'], dtype='float64'); assert sr == 8000 and x.ndim == 1 and it.get('preprocess') == 'chain8'
        sign = float(it['polarity_sign']); assert sign in (1.0, -1.0)
        return chain8(sign * x), dict(n_chain8=1, polarity_applied=1, input_sr=8000, input_len=len(x))
    assert it['condition'] in ('clean', 'reverb') and 'preprocess' not in it
    x, sr = sf.read(it['audio_path'], dtype='float32'); assert sr == 16000 and x.ndim == 1
    return x, dict(n_chain8=0, polarity_applied=0, input_sr=16000, input_len=len(x))
