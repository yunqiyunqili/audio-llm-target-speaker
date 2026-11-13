"""frontend_mix_v1 evaluation v3 (docs/frontend_mix_v1_eval_v3.md, frozen): polarity correction before direction
estimation. For each stream y_k: c_k = sum_t y_k(t) * x_0(t) (zero lag, mic-0 observation); y_k <- -y_k if c_k < 0.
No reference signal, no hard-coded stream index. Estimator / tie rule identical to v1 (eval_frontend_mix_v1.doa).
Accuracy vs direct-reference matching (v2 ref_stream_direct). Existing separated_v1 outputs only.
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/eval_frontend_mix_v1_v3.py MIX_DIR SEP_ROOT V2_DIR OUT_DIR
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
from eval_frontend_mix_v1 import BINS, DB, FRONTENDS, THRESH, doa, wrap  # noqa: E402


def polarity_fix(y, x0):
    c = (y * x0[None]).sum(1)
    return y * np.where(c < 0, -1.0, 1.0)[:, None], (c < 0)


def choose(az, th1):
    d = [abs(wrap(az[k] - th1)) for k in range(2)]
    return 0 if d[0] <= d[1] else 1


def main(mix_dir, sep_root, v2dir, out):
    mix_dir, sep_root, v2dir, out = Path(mix_dir), Path(sep_root), Path(v2dir), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(mix_dir / 'manifest.jsonl'))}
    v2 = {(r['frontend'], r['key']): r for r in map(json.loads, open(v2dir / 'eval_v2_per_sample.jsonl'))}
    geo = json.load(open(DB))['datasets']['cv_dev93']
    rows = []
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; th1, th2 = g['rir']['theta1'], g['rir']['theta2']
        mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); c = mic.mean(1)
        ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
        obs = sf.read(mix_dir / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True)[0].T
        dz = np.load(v2dir / 'direct_refs' / f'{key}.npz'); t = m['target']; i = 'B' if t == 'A' else 'A'
        for fe in FRONTENDS:
            y = np.stack([sf.read(sep_root / fe / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
            yf, flipped = polarity_fix(y, obs[0])
            az = [wrap(doa(yf[k], obs, mic[:2].T) - ra) for k in range(2)]
            v = v2[(fe, key)]; ts = v['ref_stream_direct']
            ds = choose(az, th1)
            refs = {ts: dz[t], 1 - ts: dz[i]}
            corr_after = [float(np.dot(yf[k], refs[k]) / (np.linalg.norm(yf[k]) * np.linalg.norm(refs[k]) + 1e-20)) for k in range(2)]
            rows.append(dict(key=key, group=m['group'], swap=m['swap'], frontend=fe, stratum=g['rir']['stratum'], t60=g['rir']['t60'],
                             overlap=m['overlap_active_frames'], sir_full=m['sir_mic0_full_db'], theta1=th1, theta2=th2,
                             doa_v3=az, doa_v2=v['doa'], flipped=[bool(f) for f in flipped], corr_ref_after=corr_after,
                             ref_stream_direct=ts, dir_stream_v3=ds, dir_correct_v3=ds == ts, dir_correct_v2=v['dir_correct'],
                             err_target_stream=abs(wrap(az[ts] - th1)), err_interferer_stream=abs(wrap(az[1 - ts] - th2)),
                             err_stream=[abs(wrap(az[k] - (th1 if k == ts else th2))) for k in range(2)],
                             doa_equal=az[0] == az[1], doa_within_5=abs(wrap(az[0] - az[1])) <= 5))
        print('DONE', key, flush=True)

    def summ(sel):
        if not sel:
            return dict(n=0)
        q = lambda f: float(np.mean([f(r) for r in sel]))  # noqa: E731
        return dict(n=len(sel), dir_acc_v3=q(lambda r: r['dir_correct_v3']), dir_acc_v2=q(lambda r: r['dir_correct_v2']),
                    median_err_target_stream=float(np.median([r['err_target_stream'] for r in sel])),
                    median_err_interferer_stream=float(np.median([r['err_interferer_stream'] for r in sel])),
                    median_err_stream0=float(np.median([r['err_stream'][0] for r in sel])), median_err_stream1=float(np.median([r['err_stream'][1] for r in sel])),
                    flipped_stream0=q(lambda r: r['flipped'][0]), flipped_stream1=q(lambda r: r['flipped'][1]),
                    corr_after_positive_stream0=q(lambda r: r['corr_ref_after'][0] > 0), corr_after_positive_stream1=q(lambda r: r['corr_ref_after'][1] > 0),
                    doa_equal=q(lambda r: r['doa_equal']), doa_within_5=q(lambda r: r['doa_within_5']))
    rep = dict(spec='docs/frontend_mix_v1_eval_v3.md', note='correction proposed after seeing mix_v1 results; mix_v1 numbers may be optimistic',
               threshold=THRESH, frontends={})
    for fe in FRONTENDS:
        R = [r for r in rows if r['frontend'] == fe]; G = [r for r in R if r['stratum'] != '[0,15)']
        acc = float(np.mean([r['dir_correct_v3'] for r in G]))
        strata = {'stratum': {s: summ([r for r in R if r['stratum'] == s]) for s in sorted({r['stratum'] for r in R})}}
        for k, bins in BINS.items():
            strata[k] = {f'[{lo:g},{hi:g})': summ([r for r in R if lo <= r[k] < hi]) for lo, hi in bins}
        rep['frontends'][fe] = dict(all=summ(R), excl_0_15=summ(G), dir_acc_v3_excl_0_15=acc,
                                    decision_v3='C5 allowed (recorded only; no C5 this round)' if acc >= THRESH - 1e-12 else 'not C5',
                                    failed_samples_excl_0_15=[dict(key=r['key'], stratum=r['stratum'], theta1=round(r['theta1'], 1), theta2=round(r['theta2'], 1),
                                                                   doa_v3=[round(x, 1) for x in r['doa_v3']], flipped=r['flipped'],
                                                                   ref_stream=r['ref_stream_direct'], dir_stream=r['dir_stream_v3'])
                                                              for r in G if not r['dir_correct_v3']], strata=strata)
    (out / 'eval_v3.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(out / 'eval_v3_per_sample.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    for fe, v in rep['frontends'].items():
        print(fe, json.dumps({k: v[k] for k in ('all', 'excl_0_15', 'dir_acc_v3_excl_0_15', 'decision_v3')}))
        print('   failed', len(v['failed_samples_excl_0_15']))


if __name__ == '__main__':
    main(*sys.argv[1:5])
