"""Snapshot front-ends on mix_v1: v3 evaluation + C5-style selection (docs/frontend_compare_v1.md). IN-TRAINING snapshots.
For each model: direct-reference PIT (SI-SDRi; evaluation only), v3 polarity rule + doa + choose(theta1) selection
(reference-free; oracle query azimuth as in C5), selection accuracy vs direct-reference matching (all / excl [0,15)).
Writes OUT/<model>/selection.jsonl (same format as outputs/c5_v1/selection.jsonl) and OUT/eval_snapshot_mix.json.
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/eval_snapshot_mix.py SEP_ROOT V2_DIR OUT_DIR model=snapdir ...
"""
import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frontend_mix_v1 import DB, doa, pit, sisdr, wrap  # noqa: E402
from eval_frontend_mix_v1_v3 import choose, polarity_fix  # noqa: E402

sys.path.append(str(PROJ / 'scripts/qwen'))
import split_cfg  # noqa: E402  (2026-11-12 SLURP test switch; devel default unchanged)
MIX = split_cfg.MIX


def main(sep_root, v2dir, out, *specs, final=False):
    sep_root, v2dir, out = Path(sep_root), Path(v2dir), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(split_cfg.LIST))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(MIX / 'manifest.jsonl'))}
    geo = json.load(open(DB))['datasets'][split_cfg.RIRSET]
    rep = dict(spec='docs/frontend_compare_v1.md', note=('FINAL checkpoint; selection uses oracle query azimuth (C5)' if final else 'IN-TRAINING SNAPSHOTS, diagnosis only; selection uses oracle query azimuth (C5)'), models={})
    for spec in specs:
        m, snap = spec.split('='); d = sep_root / snap; (out / m).mkdir()
        rows = []
        with open(out / m / 'selection.jsonl', 'w') as fh:
            for key in sorted(man):
                mm = man[key]; g = L[mm['group']]; t = mm['target']; i = 'B' if t == 'A' else 'A'; th1 = g['rir']['theta1']
                mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); c = mic.mean(1)
                ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
                obs = sf.read(MIX / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True)[0].T
                y = np.stack([sf.read(d / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
                yf, flipped = polarity_fix(y, obs[0]); az = [wrap(doa(yf[k], obs, mic[:2].T) - ra) for k in range(2)]
                st = choose(az, th1)
                fh.write(json.dumps(dict(key=key, stream=st, theta_query=th1, doa=az, flipped=[bool(f) for f in flipped])) + '\n')
                dz = np.load(v2dir / 'direct_refs' / f'{key}.npz')
                p, avg, per = pit([dz[t], dz[i]], y)
                ms = [sisdr(dz[t], obs[0]), sisdr(dz[i], obs[0])]
                corr = [float(np.dot(y[p[0]], dz[t]) / (np.linalg.norm(y[p[0]]) * np.linalg.norm(dz[t]) + 1e-20)),
                        float(np.dot(y[p[1]], dz[i]) / (np.linalg.norm(y[p[1]]) * np.linalg.norm(dz[i]) + 1e-20))]
                rows.append(dict(key=key, stratum=g['rir']['stratum'], sisdri_avg=avg - float(np.mean(ms)), sisdri_target=per[0] - ms[0],
                                 ref_stream=p[0], sel=st, correct=st == p[0], flipped=[bool(f) for f in flipped], corr_by_stream={p[0]: corr[0], p[1]: corr[1]},
                                 doa_within_5=abs(wrap(az[0] - az[1])) <= 5))
        G = [r for r in rows if r['stratum'] != '[0,15)']
        q = lambda sel, f: float(np.mean([f(r) for r in sel]))  # noqa: E731
        rep['models'][m] = dict(snapshot=snap, n=len(rows), direct_sisdri_avg=q(rows, lambda r: r['sisdri_avg']),
                                direct_sisdri_target=q(rows, lambda r: r['sisdri_target']), sel_acc_all=q(rows, lambda r: r['correct']),
                                sel_acc_excl_0_15=q(G, lambda r: r['correct']), flip_rate_stream0=q(rows, lambda r: r['flipped'][0]),
                                flip_rate_stream1=q(rows, lambda r: r['flipped'][1]),
                                corr_negative_stream0=q(rows, lambda r: r['corr_by_stream'][0] < 0), corr_negative_stream1=q(rows, lambda r: r['corr_by_stream'][1] < 0),
                                doa_within_5=q(rows, lambda r: r['doa_within_5']),
                                by_stratum={s: dict(n=sum(r['stratum'] == s for r in rows), sel_acc=q([r for r in rows if r['stratum'] == s], lambda r: r['correct']),
                                                    direct_sisdri_avg=q([r for r in rows if r['stratum'] == s], lambda r: r['sisdri_avg']))
                                            for s in sorted({r['stratum'] for r in rows})},
                                failed_excl_0_15=[r['key'] for r in G if not r['correct']])
        with open(out / m / 'per_sample.jsonl', 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        print(m, json.dumps({k: v for k, v in rep['models'][m].items() if k != 'failed_excl_0_15'}), flush=True)
    (out / 'eval_snapshot_mix.json').write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
