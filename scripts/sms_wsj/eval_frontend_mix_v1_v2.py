"""frontend_mix_v1 evaluation v2 (docs/frontend_mix_v1_eval_v2.md, frozen). Re-evaluates the EXISTING separated_v1
outputs only. References: direct (outputs/frontend_mix_v1_eval_v2/direct_refs, validated), early (files), full
(recomputed as in v1). Reference matching = direct-reference PIT; direction estimator and tie rule identical to v1
(eval_frontend_mix_v1.doa / pit / wrap imported unchanged). C5: dir accuracy >= 0.85 excluding [0,15).
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/eval_frontend_mix_v1_v2.py MIX_DIR SEP_ROOT SLURP_DIR V2_DIR V1_PER_SAMPLE
"""
import sys
sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import fftconvolve, resample_poly  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frontend_mix_v1 import BINS, DB, FRONTENDS, THRESH, doa, pit, sisdr, wrap  # noqa: E402
from mix_slurp_v1 import extract_piece  # noqa: E402

OFFICIAL = dict(test_eval92=dict(blstm_mvdr=4.412, fasnet_tac=14.895), cv_dev93_50=dict(blstm_mvdr=4.171, fasnet_tac=15.053))


def boot(rows, pred):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['group']].append(pred(r))
    g = sorted(by); rng = np.random.default_rng(2026)
    vals = [np.mean([v for gi in rng.choice(g, len(g)) for v in by[gi]]) for _ in range(10000)]
    return dict(p=float(np.mean([pred(r) for r in rows])), ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], n=len(rows))


def main(mix_dir, sep_root, audio_dir, v2dir, v1_per_sample):
    mix_dir, sep_root, v2dir = Path(mix_dir), Path(sep_root), Path(v2dir)
    assert json.load(open(v2dir / 'direct_refs_summary.json'))['validation']['passed']
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(mix_dir / 'manifest.jsonl'))}
    v1 = {(r['frontend'], r['key']): r for r in map(json.loads, open(v1_per_sample))}
    geo = json.load(open(DB))['datasets']['cv_dev93']
    src_cache, rows = {}, []
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; T = m['T_8k']; t = m['target']; i = 'B' if t == 'A' else 'A'
        th1, th2 = g['rir']['theta1'], g['rir']['theta2']
        mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); c = mic.mean(1)
        ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
        obs, _ = sf.read(mix_dir / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True); obs = obs.T
        dz = np.load(v2dir / 'direct_refs' / f'{key}.npz')
        refs = {}
        for k in 'AB':
            f = g[k]['file']
            if f not in src_cache:
                x, _ = sf.read(Path(audio_dir) / f, dtype='float64'); src_cache[f] = resample_poly(x, 1, 2)
            info = m['sources'][k]
            h, _ = sf.read(Path(g['rir']['rir_dir']) / f"h_{info['pos']}.wav", dtype='float64', always_2d=True)
            full = extract_piece(fftconvolve(src_cache[f], h.T[0]), g[k]['offset_8k'] - info['rir_start'], T) * info['gain']
            early = sf.read(mix_dir / key / f'early_{k}_6ch_8k.wav', dtype='float64', always_2d=True)[0][:, 0]
            refs[k] = dict(direct=dz[k], early=early, full=full)
        mix_s = {rt: [sisdr(refs[t][rt], obs[0]), sisdr(refs[i][rt], obs[0])] for rt in ('direct', 'early', 'full')}
        for fe in FRONTENDS:
            y = np.stack([sf.read(sep_root / fe / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
            res = {}
            for rt in ('direct', 'early', 'full'):
                p, avg, per = pit([refs[t][rt], refs[i][rt]], y)
                res[rt] = dict(target_stream=p[0], sisdr_avg=avg, sisdri_avg=avg - float(np.mean(mix_s[rt])),
                               sisdr_target=per[0], sisdri_target=per[0] - mix_s[rt][0])
            az = v1[(fe, key)]['doa']                       # identical estimator; recomputation would give the same values
            az_chk = wrap(doa(y[0], obs, mic[:2].T) - ra)
            assert abs(wrap(az_chk - az[0])) < 1e-9
            d = [abs(wrap(az[k] - th1)) for k in range(2)]
            dir_stream = 0 if d[0] <= d[1] else 1
            ts = res['direct']['target_stream']; sA = ts if t == 'A' else 1 - ts
            rows.append(dict(key=key, group=m['group'], swap=m['swap'], frontend=fe, stratum=g['rir']['stratum'], t60=g['rir']['t60'],
                             overlap=m['overlap_active_frames'], sir_full=m['sir_mic0_full_db'], theta1=th1, theta2=th2, doa=az,
                             ref=res, mix_sisdr=mix_s, dir_stream=dir_stream, ref_stream_direct=ts,
                             ref_stream_early=res['early']['target_stream'], dir_correct=dir_stream == ts,
                             doa_equal=az[0] == az[1], doa_within_5=abs(wrap(az[0] - az[1])) <= 5,
                             stream_A_closer_theta1=abs(wrap(az[sA] - th1)) < abs(wrap(az[sA] - th2)),
                             stream_B_closer_theta1=abs(wrap(az[1 - sA] - th1)) < abs(wrap(az[1 - sA] - th2)),
                             doa_err_target_stream=abs(wrap(az[ts] - th1)), doa_err_interferer_stream=abs(wrap(az[1 - ts] - th2))))
        print('DONE', key, flush=True)

    def summ(sel):
        if not sel:
            return dict(n=0)
        q = lambda f: float(np.mean([f(r) for r in sel]))  # noqa: E731
        out = dict(n=len(sel), dir_acc=q(lambda r: r['dir_correct']), direct_vs_early_match_agree=q(lambda r: r['ref_stream_direct'] == r['ref_stream_early']),
                   doa_equal=q(lambda r: r['doa_equal']), doa_within_5=q(lambda r: r['doa_within_5']),
                   median_doa_err_target_stream=float(np.median([r['doa_err_target_stream'] for r in sel])),
                   median_doa_err_interferer_stream=float(np.median([r['doa_err_interferer_stream'] for r in sel])))
        for rt in ('direct', 'early', 'full'):
            for k in ('sisdr_avg', 'sisdri_avg', 'sisdr_target', 'sisdri_target'):
                out[f'{k}_{rt}'] = q(lambda r, rt=rt, k=k: r['ref'][rt][k])
            out[f'mix_sisdr_{rt}'] = q(lambda r, rt=rt: np.mean(r['mix_sisdr'][rt]))
        return out
    rep = dict(spec='docs/frontend_mix_v1_eval_v2.md', threshold=THRESH, official_same_metric=OFFICIAL, frontends={})
    for fe in FRONTENDS:
        R = [r for r in rows if r['frontend'] == fe]; G = [r for r in R if r['stratum'] != '[0,15)']
        acc = float(np.mean([r['dir_correct'] for r in G]))
        strata = {'stratum': {s: summ([r for r in R if r['stratum'] == s]) for s in sorted({r['stratum'] for r in R})}}
        for k, bins in BINS.items():
            strata[k] = {f'[{lo:g},{hi:g})': summ([r for r in R if lo <= r[k] < hi]) for lo, hi in bins}
        streams = {}
        for gname, sel in [('all', R)] + [(f'stratum {s}', [r for r in R if r['stratum'] == s]) for s in sorted({r['stratum'] for r in R})] + \
                [(f'{k} [{lo:g},{hi:g})', [r for r in R if lo <= r[k] < hi]) for k, bins in BINS.items() for lo, hi in bins]:
            if sel:
                streams[gname] = dict(stream_A=boot(sel, lambda r: r['stream_A_closer_theta1']), stream_B=boot(sel, lambda r: r['stream_B_closer_theta1']))
        sm = summ(R)
        rep['frontends'][fe] = dict(all=sm, excl_0_15=summ(G), dir_acc_excl_0_15=acc,
                                    decision_v2='C5 allowed' if acc >= THRESH - 1e-12 else 'not C5',
                                    gap_to_official_direct_sisdri=dict(test_eval92=sm['sisdri_avg_direct'] - OFFICIAL['test_eval92'][fe],
                                                                       cv_dev93_50=sm['sisdri_avg_direct'] - OFFICIAL['cv_dev93_50'][fe]),
                                    failed_samples_excl_0_15=[dict(key=r['key'], stratum=r['stratum'], doa=r['doa'], theta1=r['theta1'], theta2=r['theta2'],
                                                                   dir_stream=r['dir_stream'], ref_stream_direct=r['ref_stream_direct'])
                                                              for r in G if not r['dir_correct']],
                                    strata=strata, streams_closer_theta1=streams)
    (v2dir / 'eval_v2.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(v2dir / 'eval_v2_per_sample.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    for fe, v in rep['frontends'].items():
        print(fe, json.dumps({k: v[k] for k in ('all', 'excl_0_15', 'dir_acc_excl_0_15', 'decision_v2', 'gap_to_official_direct_sisdri')}))
        print('   streams all', json.dumps(v['streams_closer_theta1']['all']))


if __name__ == '__main__':
    main(*sys.argv[1:6])
