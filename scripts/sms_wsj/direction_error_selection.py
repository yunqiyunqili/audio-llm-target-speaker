"""Direction-error selection v1 (docs/direction_error_selection_v1.md, frozen). Read-only, pure Python on existing JSON:
stream DOAs from v3 (FaSNet-TAC: outputs/c5_v1/selection.jsonl; BLSTM+MVDR: eval_v3_per_sample.jsonl doa_v3), query
theta = wrap(theta1 + delta), selection rule = frozen v3 choose(); accuracy vs v2 direct-reference matching.
Group bootstrap CI (10,000, default_rng(2026)). delta = 0 must equal the C5 selection for FaSNet-TAC.
Usage: python3 scripts/sms_wsj/direction_error_selection.py OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
DELTAS = (0, 10, -10, 20, -20)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def choose(az, th):                       # identical to eval_frontend_mix_v1_v3.choose
    d = [abs(wrap(az[k] - th)) for k in range(2)]
    return 0 if d[0] <= d[1] else 1


def stratum_t60(t):
    return '[0.2,0.3)' if t < 0.3 else '[0.3,0.4)' if t < 0.4 else '[0.4,0.5]'


def boot(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['group']].append(r['ok'])
    g = sorted(by); rng = np.random.default_rng(2026)
    vals = [np.mean([v for gi in rng.choice(g, len(g)) for v in by[gi]]) for _ in range(10000)]
    return dict(n=len(rows), acc=float(np.mean([r['ok'] for r in rows])), ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))])


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    sel = {r['key']: r for r in map(json.loads, open(PROJ / 'outputs/c5_v1/selection.jsonl'))}
    v3 = {(r['frontend'], r['key']): r for r in map(json.loads, open(PROJ / 'outputs/frontend_mix_v1_eval_v3/mix_v1/eval_v3_per_sample.jsonl'))}
    ref = {(r['frontend'], r['key']): r['ref_stream_direct'] for r in map(json.loads, open(PROJ / 'outputs/frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl'))}
    rep = dict(spec='docs/direction_error_selection_v1.md', deltas=DELTAS, frontends={})
    for fe in ('fasnet_tac', 'blstm_mvdr'):
        res = {}
        for d in DELTAS:
            rows = []
            for key in sorted(sel):
                g = L[int(key[1:4])]; th1 = g['rir']['theta1']
                az = sel[key]['doa'] if fe == 'fasnet_tac' else v3[(fe, key)]['doa_v3']
                k = choose(az, wrap(th1 + d))
                if fe == 'fasnet_tac' and d == 0:
                    assert k == sel[key]['stream'], key           # must equal C5 selection
                rows.append(dict(key=key, group=g['group'], stratum=g['rir']['stratum'], t60=stratum_t60(g['rir']['t60']), ok=k == ref[(fe, key)]))
            st = dict(all=boot(rows), excl_0_15=boot([r for r in rows if r['stratum'] != '[0,15)']))
            for s in sorted({r['stratum'] for r in rows}):
                st[f'angle {s}'] = boot([r for r in rows if r['stratum'] == s])
            for s in sorted({r['t60'] for r in rows}):
                st[f't60 {s}'] = boot([r for r in rows if r['t60'] == s])
                st[f't60 {s} excl_0_15'] = boot([r for r in rows if r['t60'] == s and r['stratum'] != '[0,15)'])
            res[str(d)] = st
        rep['frontends'][fe] = res
    a20 = min(rep['frontends']['fasnet_tac']['20']['excl_0_15']['acc'], rep['frontends']['fasnet_tac']['-20']['excl_0_15']['acc'])
    rep['decision'] = dict(acc_pm20_excl_0_15=a20, delta0_equals_C5=True,
                           verdict='direction error is not the current selection bottleneck' if a20 >= 0.95 - 1e-12 else
                           'direction uncertainty is an important bottleneck' if a20 < 0.85 - 1e-12 else 'important only under some conditions')
    (out / 'direction_error_selection.json').write_text(json.dumps(rep, indent=1) + '\n')
    for fe, res in rep['frontends'].items():
        for d, st in res.items():
            print(fe, f'delta={d:>3}', ' '.join(f"{k}:{v['acc']:.3f}[{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}](n={v['n']})" for k, v in st.items() if 'excl_0_15' not in k or k == 'excl_0_15'))
    print(json.dumps(rep['decision']))


if __name__ == '__main__':
    main(sys.argv[1])
