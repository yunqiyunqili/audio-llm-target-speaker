"""C6-delta + abstention evaluation (docs/c6_delta_v1.md, frozen). Simulated query error delta; NOT real user error.
Per delta: v3 choose() on stored stream DOAs with theta_query = wrap(theta1 + delta); downstream output = C5 raw (stream
chosen by C5) or OTHER raw (the other stream); eval_c4_v1.score / summ unchanged. Abstention: margin = |q - doa_other| -
|q - doa_chosen| < tau -> abstain; groups split 50/50 by random.Random(2026); tau chosen on dev (pooled deltas), reported on eval.
Usage: python -I scripts/qwen/eval_c6_delta.py C5_RAW OTHER_RAW SELECTION V2_PER_SAMPLE MIX_MANIFEST OUT_DIR
"""
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402

DELTAS = (0, 10, -10, 20, -20)
NATURAL = {'[0,15)': 78, '[15,30)': 96, '[30,60)': 155, '[60,180]': 653}
TAUS = list(range(0, 91))
KEEP = ('n', 'only_target', 'only_interferer', 'both', 'neither', 'target_any', 'target_all', 'pair_flip_accuracy', 'M', 'U')


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def choose(az, th):
    d = [abs(wrap(az[k] - th)) for k in range(2)]
    return 0 if d[0] <= d[1] else 1


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(c5raw, otherraw, selp, v2ps, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    R5, RO = load(c5raw), load(otherraw)
    sel = {r['key']: r for r in map(json.loads, open(selp))}
    ref = {r['key']: r['ref_stream_direct'] for r in map(json.loads, open(v2ps)) if r['frontend'] == 'fasnet_tac'}
    for k in sel:
        assert R5[k]['stream'] == sel[k]['stream'] and RO[k]['stream'] == 1 - sel[k]['stream']
    pick = lambda s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k in KEEP}  # noqa: E731
    gids = sorted(L); random.Random(2026).shuffle(gids); dev, ev = set(gids[:50]), set(gids[50:])
    rep = dict(spec='docs/c6_delta_v1.md', note='simulated fixed query offsets; not real user or localiser error; delta=0 is C5 (oracle azimuth)',
               natural_weights=NATURAL, dev_groups=sorted(dev), eval_groups=sorted(ev), deltas={})
    per = {}
    for d in DELTAS:
        raws, info = {}, {}
        for k in sorted(sel):
            q = wrap(L[int(k[1:4])]['rir']['theta1'] + d); az = sel[k]['doa']; c = choose(az, q)
            raws[k] = R5[k] if c == sel[k]['stream'] else RO[k]
            info[k] = dict(chosen=c, margin=abs(wrap(q - az[1 - c])) - abs(wrap(q - az[c])), sel_wrong=c != ref[k])
        rows = {r['key']: r for r in score(raws, L, man)}
        per[d] = (rows, info)
        strata = {s: summ([r for r in rows.values() if r['stratum'] == s]) for s in NATURAL}
        W = sum(NATURAL.values())
        weighted = {m: sum(NATURAL[s] / W * strata[s][m] for s in NATURAL) for m in ('only_target', 'only_interferer', 'neither', 'target_any', 'target_all')}
        rep['deltas'][str(d)] = dict(all=pick(summ(list(rows.values()))), selection_error=sum(i['sel_wrong'] for i in info.values()) / len(info),
                                     strata={s: pick(v) for s, v in strata.items()}, natural_weighted=weighted)
    c5_all = summ(score(R5, L, man))
    d0 = summ(list(per[0][0].values()))
    assert all(abs(c5_all[m] - d0[m]) < 1e-12 for m in ('target_any', 'target_all', 'only_target', 'only_interferer', 'neither')), (c5_all, d0)
    rep['delta0_equals_C5'] = True

    def answered_stats(keys_by_d, tau):
        a = [(d, k) for d, ks in keys_by_d.items() for k in ks if per[d][1][k]['margin'] >= tau]
        n = sum(len(ks) for ks in keys_by_d.values())
        if not a:
            return dict(coverage=0.0, answered=0)
        rows = [per[d][0][k] for d, k in a]
        return dict(coverage=len(a) / n, answered=len(a), sel_error=sum(per[d][1][k]['sel_wrong'] for d, k in a) / len(a),
                    only_interferer=sum(r['cat'] == 'only_interferer' for r in rows) / len(a), target_any=sum(r['t_any'] for r in rows) / len(a))
    devkeys = {d: [k for k in sel if int(k[1:4]) in dev] for d in DELTAS}
    curve_dev = {t: answered_stats(devkeys, t) for t in TAUS}
    ok = [t for t in TAUS if curve_dev[t]['answered'] and curve_dev[t]['sel_error'] <= 0.05 + 1e-12]
    tau = min(ok) if ok else min(TAUS, key=lambda t: (curve_dev[t].get('sel_error', 1.0), t))
    rep['abstention'] = dict(tau_deg=tau, tau_rule='smallest tau with dev selection error <= 0.05 (pooled deltas)', found_by_rule=bool(ok),
                             dev_at_tau=curve_dev[tau], eval={}, risk_coverage_eval={})
    for d in DELTAS:
        evk = {d: [k for k in sel if int(k[1:4]) in ev]}
        rep['abstention']['eval'][str(d)] = dict(at_tau=answered_stats(evk, tau), no_abstention=answered_stats(evk, 0))
        rep['abstention']['risk_coverage_eval'][str(d)] = [dict(tau=t, **answered_stats(evk, t)) for t in TAUS]
    (out / 'eval_c6_delta.json').write_text(json.dumps(rep, indent=1) + '\n')
    for d, v in rep['deltas'].items():
        print('delta', d, 'sel_err', round(v['selection_error'], 3), json.dumps(v['all']), 'weighted', json.dumps({k: round(x, 3) for k, x in v['natural_weighted'].items()}))
    print('TAU', tau, 'dev', json.dumps(curve_dev[tau]))
    for d, v in rep['abstention']['eval'].items():
        print('eval delta', d, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v['at_tau'].items()}), '| no-abst',
              json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v['no_abstention'].items()}))


if __name__ == '__main__':
    main(*sys.argv[1:7])
