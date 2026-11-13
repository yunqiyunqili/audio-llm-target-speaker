"""Separation-quality metrics for the C5 FaSNet-TAC target stream (docs/separation_quality_diag_v1.md, frozen; read-only).
Target stream = C5 selection (outputs/c5_v1/selection.jsonl) with its polarity applied. References = validated direct refs.
SI-SDR_T (tfgridnet sisdr), and a least-squares split y = a d_T + b d_I + e_r (zero-mean) -> IRR = 10log10(|e_t|^2/|e_i|^2),
RAR = 10log10(|e_t|^2/|e_r|^2). Writes OUT_DIR/metrics.jsonl.
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/sep_quality_metrics.py SEP_ROOT V2_DIR OUT_DIR
"""
import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frontend_mix_v1 import sisdr  # noqa: E402


def db(a, b):
    return float(10 * np.log10(max(a, 1e-30) / max(b, 1e-30)))


def main(sep_root, v2dir, out):
    sep_root, v2dir, out = Path(sep_root), Path(v2dir), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    sel = [json.loads(l) for l in open(PROJ / 'outputs/c5_v1/selection.jsonl')]
    with open(out / 'metrics.jsonl', 'w') as fh:
        for s in sel:
            key = s['key']; g = L[int(key[1:4])]; t = 'A' if key.endswith('s1') else 'B'; i = 'B' if t == 'A' else 'A'
            k = s['stream']
            y = sf.read(sep_root / 'fasnet_tac' / key / f'out{k}.wav', dtype='float64')[0] * (-1.0 if s['flipped'][k] else 1.0)
            dz = np.load(v2dir / 'direct_refs' / f'{key}.npz'); dT, dI = dz[t].astype(np.float64), dz[i].astype(np.float64)
            yc, Tc, Ic = y - y.mean(), dT - dT.mean(), dI - dI.mean()
            coef, *_ = np.linalg.lstsq(np.stack([Tc, Ic], 1), yc, rcond=None)
            et, ei = coef[0] * Tc, coef[1] * Ic; er = yc - et - ei
            fh.write(json.dumps(dict(key=key, group=g['group'], stream=k, sisdr_T=sisdr(dT, y), irr_db=db((et ** 2).sum(), (ei ** 2).sum()),
                                     rar_db=db((et ** 2).sum(), (er ** 2).sum()), coef_T=float(coef[0]), coef_I=float(coef[1]))) + '\n')
    print('WROTE', len(sel), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
