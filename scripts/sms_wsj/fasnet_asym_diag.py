"""FaSNet-TAC direction asymmetry diagnostic (docs/c4_v1.md; read-only; no new front-end output; no fix).
Estimator = eval_frontend_mix_v1.doa (unchanged). Streams assigned by direct-reference matching (eval v2 per-sample).
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/fasnet_asym_diag.py MIX_DIR SEP_ROOT V2_DIR OUT_JSON
"""
import sys
sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frontend_mix_v1 import DB, doa, sisdr, wrap  # noqa: E402

MAXLAG = 400


def lag_corr(y, r):
    n = len(y); Y = np.fft.rfft(y, 2 * n); R = np.fft.rfft(r, 2 * n)
    cc = np.fft.irfft(Y * np.conj(R), 2 * n)
    cc = np.concatenate([cc[-MAXLAG:], cc[:MAXLAG + 1]])        # lags -MAXLAG..MAXLAG (y delayed vs r -> positive)
    k = int(np.argmax(np.abs(cc)))
    return k - MAXLAG, float(cc[k] / (np.linalg.norm(y) * np.linalg.norm(r) + 1e-20))


def med(x):
    return float(np.median(x)) if len(x) else None


def main(mix_dir, sep_root, v2dir, out_json):
    mix_dir, sep_root, v2dir = Path(mix_dir), Path(sep_root), Path(v2dir)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(mix_dir / 'manifest.jsonl'))}
    v2 = {r['key']: r for r in map(json.loads, open(v2dir / 'eval_v2_per_sample.jsonl')) if r['frontend'] == 'fasnet_tac'}
    geo = json.load(open(DB))['datasets']['cv_dev93']
    rows, streams = [], []
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'
        th = {'A': g['rir']['theta1'] if m['swap'] == 1 else g['rir']['theta2'], 'B': g['rir']['theta2'] if m['swap'] == 1 else g['rir']['theta1']}
        mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); c = mic.mean(1)
        ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
        obs = sf.read(mix_dir / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True)[0].T
        dz = np.load(v2dir / 'direct_refs' / f'{key}.npz'); D = {'A': dz['A'], 'B': dz['B']}
        y = np.stack([sf.read(sep_root / 'fasnet_tac' / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
        est = lambda s: wrap(doa(s, obs, mic[:2].T) - ra)  # noqa: E731
        dA, dB = est(D['A']), est(D['B'])
        r = dict(key=key, group=m['group'], swap=m['swap'], stratum=g['rir']['stratum'], theta1=g['rir']['theta1'], theta2=g['rir']['theta2'],
                 direct_err_theta1=abs(wrap((dA if th['A'] == g['rir']['theta1'] else dB) - g['rir']['theta1'])),
                 direct_err_theta2=abs(wrap((dB if th['B'] == g['rir']['theta2'] else dA) - g['rir']['theta2'])))
        ts = v2[key]['ref_stream_direct']; az = v2[key]['doa']
        owner = {ts: t, 1 - ts: i}
        for k in range(2):
            spk = owner[k]; other = 'B' if spk == 'A' else 'A'
            lag, corr = lag_corr(y[k], D[spk]); lag_o, corr_o = lag_corr(y[k], D[other])
            streams.append(dict(key=key, group=m['group'], swap=m['swap'], stream=k, speaker=spk, role='target' if k == ts else 'interferer',
                                position='theta1' if th[spk] == g['rir']['theta1'] else 'theta2', stratum=g['rir']['stratum'],
                                doa_err=abs(wrap(az[k] - th[spk])), lag_own=lag, corr_own=corr, lag_other=lag_o, corr_other=corr_o,
                                sisdr_own=sisdr(D[spk], y[k]), sisdr_other=sisdr(D[other], y[k]), rms=float(np.sqrt(np.mean(y[k] ** 2)))))
        rows.append(r)

    def grp(f):
        out = {}
        bucket = collections.defaultdict(list)
        for s in streams:
            bucket[f(s)].append(s)
        for k, v in sorted(bucket.items(), key=lambda kv: str(kv[0])):
            out[str(k)] = dict(n=len(v), median_doa_err=med([s['doa_err'] for s in v]), frac_err_le_10=float(np.mean([s['doa_err'] <= 10 for s in v])),
                               median_abs_lag_own=med([abs(s['lag_own']) for s in v]), median_lag_own=med([s['lag_own'] for s in v]),
                               median_corr_own=med([s['corr_own'] for s in v]), median_sisdr_own=med([s['sisdr_own'] for s in v]),
                               median_corr_other=med([s['corr_other'] for s in v]))
        return out
    by = dict(stream_index=grp(lambda s: s['stream']), swap=grp(lambda s: f"s{s['swap']}"), speaker=grp(lambda s: s['speaker']),
              position=grp(lambda s: s['position']), role=grp(lambda s: s['role']),
              position_x_stratum=grp(lambda s: f"{s['position']} {s['stratum']}"), stream_x_position=grp(lambda s: f"stream{s['stream']} {s['position']}"))
    direct = dict(median_err_theta1=med([r['direct_err_theta1'] for r in rows]), median_err_theta2=med([r['direct_err_theta2'] for r in rows]),
                  frac_le_10_theta1=float(np.mean([r['direct_err_theta1'] <= 10 for r in rows])),
                  frac_le_10_theta2=float(np.mean([r['direct_err_theta2'] <= 10 for r in rows])))
    tgt = sorted([s for s in streams if s['role'] == 'target'], key=lambda s: s['doa_err'])
    pick = lambda v: [{k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()} for s in v]  # noqa: E731
    # ---- judgement (rules in docs/c4_v1.md, in order)
    split = lambda a, b, hi, lo: (a > hi and b <= lo) or (b > hi and a <= lo)  # noqa: E731
    p1, p2 = by['position']['theta1'], by['position']['theta2']; s0, s1 = by['stream_index']['0'], by['stream_index']['1']
    if split(direct['median_err_theta1'], direct['median_err_theta2'], 10, 10):
        verdict = 'evaluation definition'
    elif split(s0['median_doa_err'], s1['median_doa_err'], 30, 10) and not split(p1['median_doa_err'], p2['median_doa_err'], 30, 10):
        verdict = 'output stream index'
    elif split(p1['median_doa_err'], p2['median_doa_err'], 30, 10) and (
            (lambda bad, good: bad['median_abs_lag_own'] >= 2 or bad['median_corr_own'] <= good['median_corr_own'] - 0.2)(
                *((p1, p2) if p1['median_doa_err'] > p2['median_doa_err'] else (p2, p1)))):
        verdict = 'time or phase problem'
    else:
        verdict = 'undetermined'
    rep = dict(spec='docs/c4_v1.md (FaSNet asymmetry diagnostic)', n_samples=len(rows), direct_reference_doa=direct, groups=by,
               target_stream_largest_err_10=pick(tgt[-10:][::-1]), target_stream_smallest_err_10=pick(tgt[:10]), verdict=verdict)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if 'err_10' not in k}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
