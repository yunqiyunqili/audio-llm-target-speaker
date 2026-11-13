"""C5 residual decomposition (docs/c5_residual_diag_v1.md, frozen; read-only; descriptive, not causal).
Target stream = C5 selection with polarity applied (8 kHz). Per source: direct d, early reflections r_e = early - d,
late reverberation r_l = full - early (full image recomputed in memory as in eval v2). Nested zero-mean LS residual
energies R1 (B1={d_T,d_I}), R2 (+{r_e,T, r_e,I}), R3 (+{r_l,T, r_l,I}); also {d, r_e} and {d, r_l} alone.
Exports 10 listening direct references (random.Random(2026).sample) as wav to OUT/listen (outputs only, never data/).
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/c5_residual_diag.py MIX_DIR SEP_ROOT SLURP_DIR V2_DIR C5_RAW OUT_DIR
"""
import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import fftconvolve, resample_poly  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from mix_slurp_v1 import extract_piece  # noqa: E402


def resid(y, basis):
    B = np.stack([b - b.mean() for b in basis], 1); yc = y - y.mean()
    c, *_ = np.linalg.lstsq(B, yc, rcond=None)
    return float(((yc - B @ c) ** 2).sum())


def q(x):
    return dict(p25=float(np.percentile(x, 25)), median=float(np.median(x)), p75=float(np.percentile(x, 75)))


def main(mix_dir, sep_root, slurp_dir, v2dir, c5raw, out):
    mix_dir, sep_root, v2dir, out = Path(mix_dir), Path(sep_root), Path(v2dir), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(mix_dir / 'manifest.jsonl'))}
    sel = {r['key']: r for r in map(json.loads, open(PROJ / 'outputs/c5_v1/selection.jsonl'))}
    from eval_c4_v1 import score  # noqa: E402  (pure python; imports slu code)
    hit = {r['key']: r['t_any'] for r in score({json.loads(l)['file']: json.loads(l) for l in open(c5raw)}, L, man)}
    cache, rows = {}, []
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; T = m['T_8k']; t = m['target']; i = 'B' if t == 'A' else 'A'; s = sel[key]; k = s['stream']
        y = sf.read(sep_root / 'fasnet_tac' / key / f'out{k}.wav', dtype='float64')[0] * (-1.0 if s['flipped'][k] else 1.0)
        dz = np.load(v2dir / 'direct_refs' / f'{key}.npz')
        comp = {}
        for spk in 'AB':
            f = g[spk]['file']
            if f not in cache:
                cache[f] = resample_poly(sf.read(Path(slurp_dir) / f, dtype='float64')[0], 1, 2)
            info = m['sources'][spk]
            h = sf.read(Path(g['rir']['rir_dir']) / f"h_{info['pos']}.wav", dtype='float64', always_2d=True)[0].T
            full = extract_piece(fftconvolve(cache[f], h[0]), g[spk]['offset_8k'] - info['rir_start'], T) * info['gain']
            early = sf.read(mix_dir / key / f'early_{spk}_6ch_8k.wav', dtype='float64', always_2d=True)[0][:, 0]
            d = dz[spk].astype(np.float64)
            comp[spk] = dict(d=d, re=early - d, rl=full - early)
        D = [comp[t]['d'], comp[i]['d']]; E = [comp[t]['re'], comp[i]['re']]; Lr = [comp[t]['rl'], comp[i]['rl']]
        R1 = resid(y, D); R2 = resid(y, D + E); R3 = resid(y, D + E + Lr); Re = R2; Rl = resid(y, D + Lr)
        rows.append(dict(key=key, early_explained=(R1 - R2) / R1, late_explained=(R2 - R3) / R1, unexplained=R3 / R1,
                         early_alone=(R1 - Re) / R1, late_alone=(R1 - Rl) / R1, c5_target_hit=bool(hit[key])))
    rep = dict(spec='docs/c5_residual_diag_v1.md', note='descriptive LS decomposition; components are correlated; not causal', n=len(rows))
    for name, sel_rows in (('all', rows), ('c5_hit', [r for r in rows if r['c5_target_hit']]), ('c5_miss', [r for r in rows if not r['c5_target_hit']])):
        rep[name] = dict(n=len(sel_rows), **{f: q([r[f] for r in sel_rows]) for f in ('early_explained', 'late_explained', 'unexplained', 'early_alone', 'late_alone')})
    listen = random.Random(2026).sample(sorted(man), 10)
    (out / 'listen').mkdir()
    rep['listening'] = []
    for key in listen:
        m = man[key]; t = m['target']; k = sel[key]['stream']
        p = out / 'listen' / f'{key}_direct_target_8k.wav'
        sf.write(p, np.load(v2dir / 'direct_refs' / f'{key}.npz')[t].astype(np.float32), 8000, subtype='FLOAT')
        rep['listening'].append(dict(key=key, target=t, c5_target_stream=str(sep_root / 'fasnet_tac' / key / f'out{k}.wav'),
                                     direct_reference=str(p.resolve()), c5_target_hit=bool(hit[key]),
                                     sentence=L[m['group']][t]['sentence']))
    (out / 'residual.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(out / 'per_sample.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'listening'}, indent=1))
    for x in rep['listening']:
        print('LISTEN', json.dumps(x))


if __name__ == '__main__':
    main(*sys.argv[1:7])
