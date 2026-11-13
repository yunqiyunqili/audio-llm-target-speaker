"""Front-end sanity check D (theta1/theta2 asymmetry; read-only) + final decision per docs/frontend_sanity_v1.md.
Reads mix_v1 (obs, early images, manifest), separated_v1 outputs, the 22014183 evaluation per_sample.jsonl and the
A/B/C JSONs. Direction estimator = eval_frontend_mix_v1.doa (unchanged). CI: group bootstrap, 10,000, default_rng(2026).
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/frontend_sanity_d.py MIX_DIR SEP_ROOT EVAL_PER_SAMPLE SANITY_DIR OUT_JSON
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
from eval_frontend_mix_v1 import DB, doa, wrap  # noqa: E402

FRONTENDS = ('blstm_mvdr', 'fasnet_tac')


def closer(az, t1, t2):
    d1, d2 = abs(wrap(az - t1)), abs(wrap(az - t2))
    return 'theta1' if d1 < d2 else 'theta2' if d2 < d1 else 'tie'


def boot(rows, pred):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['group']].append(pred(r))
    g = sorted(by); rng = np.random.default_rng(2026); vals = []
    for _ in range(10000):
        s = [v for gi in rng.choice(g, len(g)) for v in by[gi]]
        vals.append(np.mean(s))
    p = float(np.mean([pred(r) for r in rows]))
    return dict(p=p, ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], n=len(rows))


def main(mix_dir, sep_root, per_sample, sanity_dir, out_json):
    mix_dir, sep_root = Path(mix_dir), Path(sep_root)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(mix_dir / 'manifest.jsonl'))}
    ev = {(r['frontend'], r['key']): r for r in map(json.loads, open(per_sample))}
    geo = json.load(open(DB))['datasets']['cv_dev93']
    rows, struct = [], collections.Counter()
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; T = m['T_8k']; t1, t2 = g['rir']['theta1'], g['rir']['theta2']
        mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); c = mic.mean(1)
        ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
        obs, sr = sf.read(mix_dir / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True)
        struct['obs_ok'] += sr == 8000 and obs.shape == (T, 6)
        eA, _ = sf.read(mix_dir / key / 'early_A_6ch_8k.wav', dtype='float64', always_2d=True)
        eB, _ = sf.read(mix_dir / key / 'early_B_6ch_8k.wav', dtype='float64', always_2d=True)
        clean = eA + eB
        cc = np.array([[np.corrcoef(obs[:, i], clean[:, j])[0, 1] for j in range(6)] for i in range(6)])
        struct['channel_order_ok'] += bool(np.all(cc.argmax(1) == np.arange(6)))
        D = lambda y: wrap(doa(y, obs.T, mic[:2].T) - ra)  # noqa: E731
        az_obs, az_A, az_B = D(obs[:, 0]), D(eA[:, 0]), D(eB[:, 0])
        posA = 'theta1' if m['swap'] == 1 else 'theta2'; posB = 'theta2' if m['swap'] == 1 else 'theta1'
        r = dict(key=key, group=m['group'], swap=m['swap'], stratum=g['rir']['stratum'], theta1=t1, theta2=t2,
                 obs=closer(az_obs, t1, t2), oracle_A=closer(az_A, t1, t2), oracle_B=closer(az_B, t1, t2),
                 oracle_A_own=closer(az_A, t1, t2) == posA, oracle_B_own=closer(az_B, t1, t2) == posB, fe={})
        for fe in FRONTENDS:
            y = np.stack([sf.read(sep_root / fe / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
            struct[f'{fe}_out_ok'] += y.shape == (2, T)
            e = ev[(fe, key)]; ts = e['ref_stream']; t = m['target']
            sA = ts if t == 'A' else 1 - ts
            az = e['doa']
            rms = np.sqrt((y ** 2).mean(1))
            near = lambda a_: 'theta1' if abs(wrap(a_ - t1)) <= 10 else 'theta2' if abs(wrap(a_ - t2)) <= 10 else 'elsewhere'  # noqa: E731
            r['fe'][fe] = dict(stream_A=closer(az[sA], t1, t2), stream_B=closer(az[1 - sA], t1, t2),
                               target_stream_near=near(az[ts]), interferer_stream_near=near(az[1 - ts]),
                               both_streams_same_side=closer(az[0], t1, t2) == closer(az[1], t1, t2),
                               rms_ratio=float(rms.min() / max(rms.max(), 1e-20)), near_silent=bool(rms.min() < 1e-4 * rms.max()),
                               target_is_louder_stream=bool(rms[ts] >= rms[1 - ts]), dir_correct=e['dir_correct'])
        rows.append(r)
    n = len(rows)
    rep = dict(n=n, structure=dict(obs_shape_sr_ok=struct['obs_ok'], channel_order_ok=struct['channel_order_ok'],
                                   **{f'{fe}_output_shape_ok': struct[f'{fe}_out_ok'] for fe in FRONTENDS}))
    rep['structure_pass'] = all(v == n for v in rep['structure'].values())
    rep['oracle_own_position'] = dict(A=boot(rows, lambda r: r['oracle_A_own']), B=boot(rows, lambda r: r['oracle_B_own']),
                                      both=float(np.mean([r['oracle_A_own'] and r['oracle_B_own'] for r in rows])))
    rep['oracle_pass'] = rep['oracle_own_position']['A']['p'] >= 0.95 and rep['oracle_own_position']['B']['p'] >= 0.95
    def closer1(sel, f):  # noqa: E306
        return boot(sel, lambda r: f(r) == 'theta1')
    groupings = dict(all=lambda r: True, s1=lambda r: r['swap'] == 1, s2=lambda r: r['swap'] == 2,
                     **{f'stratum_{s}': (lambda s: lambda r: r['stratum'] == s)(s) for s in sorted({r['stratum'] for r in rows})})
    rep['closer_to_theta1'] = {gname: dict(obs=closer1([r for r in rows if gf(r)], lambda r: r['obs']),
                                           oracle_A=closer1([r for r in rows if gf(r)], lambda r: r['oracle_A']),
                                           oracle_B=closer1([r for r in rows if gf(r)], lambda r: r['oracle_B']),
                                           **{f'{fe}_stream_{k}': closer1([r for r in rows if gf(r)], lambda r, fe=fe, k=k: r['fe'][fe][f'stream_{k}'])
                                              for fe in FRONTENDS for k in 'AB'}) for gname, gf in groupings.items()}
    rep['frontend_detail'] = {}
    for fe in FRONTENDS:
        F = [r['fe'][fe] for r in rows]
        rep['frontend_detail'][fe] = dict(
            target_stream_near=dict(collections.Counter(x['target_stream_near'] for x in F)),
            interferer_stream_near=dict(collections.Counter(x['interferer_stream_near'] for x in F)),
            both_streams_same_side=sum(x['both_streams_same_side'] for x in F), near_silent=sum(x['near_silent'] for x in F),
            rms_ratio_median=float(np.median([x['rms_ratio'] for x in F])),
            target_is_louder_stream=sum(x['target_is_louder_stream'] for x in F),
            by_stratum={s: dict(n=sum(r['stratum'] == s for r in rows), dir_acc=float(np.mean([r['fe'][fe]['dir_correct'] for r in rows if r['stratum'] == s])),
                                target_near=dict(collections.Counter(r['fe'][fe]['target_stream_near'] for r in rows if r['stratum'] == s)),
                                interferer_near=dict(collections.Counter(r['fe'][fe]['interferer_stream_near'] for r in rows if r['stratum'] == s)))
                        for s in sorted({r['stratum'] for r in rows})})
    rep['D_bug'] = not (rep['structure_pass'] and rep['oracle_pass'])
    # ---- final decision (A -> B -> D; C explains duration)
    sd = Path(sanity_dir); S = {fe: json.load(open(sd / f'sanity_{fe}.json')) for fe in FRONTENDS}
    mixv1 = dict(blstm_mvdr=2.8956618452226586, fasnet_tac=0.4745189993860654)   # 22014183, avg SI-SDRi, early ref
    dur = {}
    for fe in FRONTENDS:
        Cc = S[fe]['C']; gap = Cc['full_early_sisdri'] - mixv1[fe]; drop = Cc['full_early_sisdri'] - Cc['trunc_early_sisdri']
        dur[fe] = dict(gap=gap, drop=drop, judged=gap > 2.0, short_is_main=bool(gap > 2.0 and drop >= 0.5 * gap))
    A_bug = any(S[fe]['A']['problem'] for fe in FRONTENDS); B_bug = any(S[fe]['B']['problem'] for fe in FRONTENDS)
    if A_bug or rep['D_bug']:
        concl = 'existing pipeline bug'
    elif B_bug:
        concl = 'mixing script or level bug'
    else:
        concl = 'pipeline OK; mainly short utterances or SLURP domain difference'
    rep['decision'] = dict(A_problem={fe: S[fe]['A']['problem'] for fe in FRONTENDS}, B_problem={fe: S[fe]['B']['problem'] for fe in FRONTENDS},
                           D_bug=rep['D_bug'], duration=dur, conclusion=concl)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
