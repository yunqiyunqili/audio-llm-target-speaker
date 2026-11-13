"""C6-delta + abstention for the SpatialNet snapshot (docs/c6_delta_spatialnet_v1.md, frozen). Rules identical to
eval_c6_delta.py (choose / wrap / DELTAS / NATURAL / TAUS imported); differences: reference matching from the SpatialNet
evaluation (per_sample.jsonl ref_stream), two downstream routes (P2, Whisper+T2), plus T60 strata. Simulated query offsets
only; NOT real C6. IN-TRAINING snapshot, diagnosis only.
Usage: python -I scripts/qwen/eval_c6_delta_spatialnet.py SEL_RUN OTHER_RUN SELECTION SNAP_PER_SAMPLE MIX_MANIFEST FASNET_C6_JSON OUT_DIR
"""
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402
from eval_c6_delta import DELTAS, KEEP, NATURAL, TAUS, choose, load, wrap  # noqa: E402

T60 = (('[0.2,0.3)', 0.2, 0.3), ('[0.3,0.4)', 0.3, 0.4), ('[0.4,0.5]', 0.4, 0.5000001))
ROUTES = dict(P2='P2', WT2='T2')


def main(selrun, othrun, selp, snap_ps, manifest, fasnet_json, out, route='both'):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    sel = {r['key']: r for r in map(json.loads, open(selp))}
    ref = {r['key']: r['ref_stream'] for r in map(json.loads, open(snap_ps))}
    routes = ROUTES if route == 'both' else {'WT2': 'T2'}
    R = {rt: (load(Path(selrun) / d / 'raw.jsonl'), load(Path(othrun) / d / 'raw.jsonl')) for rt, d in routes.items()}
    for rt, (R5, RO) in R.items():
        for k in sel:
            assert R5[k]['file'] == k and RO[k]['file'] == k
    if 'P2' in R and 'stream' in next(iter(R['P2'][0].values())):
        for k in sel:
            assert R['P2'][0][k]['stream'] == sel[k]['stream'] and R['P2'][1][k]['stream'] == 1 - sel[k]['stream']
    pick = lambda s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k in KEEP}  # noqa: E731
    gids = sorted(L); random.Random(2026).shuffle(gids); dev, ev = set(gids[:50]), set(gids[50:])
    rep = dict(spec='docs/c6_delta_spatialnet_v1.md', route=route, note=('SpatialNet ep100 final; T2-LoRA route' if route != 'both' else 'SpatialNet ep72 IN-TRAINING snapshot, diagnosis only') + '; simulated fixed query offsets; NOT real C6; delta=0 is C5-like (oracle azimuth)',
               natural_weights=NATURAL, dev_groups=sorted(dev), eval_groups=sorted(ev), routes={})
    info = {}
    for d in DELTAS:
        info[d] = {}
        for k in sorted(sel):
            q = wrap(L[int(k[1:4])]['rir']['theta1'] + d); az = sel[k]['doa']; c = choose(az, q)
            if d == 0:
                assert c == sel[k]['stream'], k
            info[d][k] = dict(chosen=c, margin=abs(wrap(q - az[1 - c])) - abs(wrap(q - az[c])), sel_wrong=c != ref[k])
    per = {}
    for rt, (R5, RO) in R.items():
        res = {}; per[rt] = {}
        for d in DELTAS:
            raws = {k: (R5[k] if info[d][k]['chosen'] == sel[k]['stream'] else RO[k]) for k in sel}
            rows = {r['key']: r for r in score(raws, L, man)}; per[rt][d] = rows
            strata = {s: summ([r for r in rows.values() if r['stratum'] == s]) for s in NATURAL}
            W = sum(NATURAL.values())
            weighted = {m: sum(NATURAL[s] / W * strata[s][m] for s in NATURAL) for m in ('only_target', 'only_interferer', 'neither', 'target_any', 'target_all')}
            t60 = {name: pick(summ([r for r in rows.values() if lo <= L[int(r['key'][1:4])]['rir']['t60'] < hi])) for name, lo, hi in T60}
            res[str(d)] = dict(all=pick(summ(list(rows.values()))), selection_error=sum(i['sel_wrong'] for i in info[d].values()) / len(info[d]),
                               strata={s: pick(v) for s, v in strata.items()}, natural_weighted=weighted, t60=t60)
        d0 = summ(list(per[rt][0].values())); c5 = summ(score(R5, L, man))
        assert all(abs(d0[m] - c5[m]) < 1e-12 for m in ('target_any', 'target_all', 'only_target', 'only_interferer', 'neither'))
        rep['routes'][rt] = dict(deltas=res, delta0_equals_selected_run=True)

    def answered(keys_by_d, tau, rt):
        a = [(d, k) for d, ks in keys_by_d.items() for k in ks if info[d][k]['margin'] >= tau]
        n = sum(len(ks) for ks in keys_by_d.values())
        if not a:
            return dict(coverage=0.0, answered=0)
        o = dict(coverage=len(a) / n, answered=len(a), sel_error=sum(info[d][k]['sel_wrong'] for d, k in a) / len(a))
        if rt:
            rows = [per[rt][d][k] for d, k in a]
            o.update(only_interferer=sum(r['cat'] == 'only_interferer' for r in rows) / len(a), target_any=sum(r['t_any'] for r in rows) / len(a))
        return o
    devk = {d: [k for k in sel if int(k[1:4]) in dev] for d in DELTAS}
    cdev = {t: answered(devk, t, None) for t in TAUS}
    ok = [t for t in TAUS if cdev[t]['answered'] and cdev[t]['sel_error'] <= 0.05 + 1e-12]
    tau = min(ok) if ok else min(TAUS, key=lambda t: (cdev[t].get('sel_error', 1.0), t))
    ab = dict(tau_deg=tau, found_by_rule=bool(ok), dev_at_tau=cdev[tau], eval={}, risk_coverage_eval={})
    for d in DELTAS:
        evk = {d: [k for k in sel if int(k[1:4]) in ev]}
        ab['eval'][str(d)] = {rt: dict(at_tau=answered(evk, tau, rt), no_abstention=answered(evk, 0, rt)) for rt in R}
        risk_rt = 'P2' if 'P2' in R else next(iter(R))
        ab['risk_coverage_eval'][str(d)] = [dict(tau=t, **answered(evk, t, risk_rt)) for t in TAUS]
    rep['abstention'] = ab
    fas = json.load(open(fasnet_json))
    if route == 'both':
      rep['comparison'] = {str(d): dict(spatialnet_P2=rep['routes']['P2']['deltas'][str(d)]['all']['target_any'],
                                      spatialnet_WT2=rep['routes']['WT2']['deltas'][str(d)]['all']['target_any'],
                                      gap_P2_minus_WT2=rep['routes']['P2']['deltas'][str(d)]['all']['target_any'] - rep['routes']['WT2']['deltas'][str(d)]['all']['target_any'],
                                      fasnet_P2=fas['deltas'][str(d)]['all']['target_any'], fasnet_P2_selection_error=fas['deltas'][str(d)]['selection_error'],
                                      spatialnet_selection_error=rep['routes']['P2']['deltas'][str(d)]['selection_error'],
                                      spatialnet_P2_weighted=rep['routes']['P2']['deltas'][str(d)]['natural_weighted']['target_any'],
                                      fasnet_P2_weighted=fas['deltas'][str(d)]['natural_weighted']['target_any']) for d in DELTAS}
    (out / 'eval_c6_delta_spatialnet.json').write_text(json.dumps(rep, indent=1) + '\n')
    # file in the format expected by plot_c6_risk_coverage.py
    (out / 'for_plot.json').write_text(json.dumps(dict(abstention=dict(tau_deg=tau, risk_coverage_eval=ab['risk_coverage_eval'])), indent=1) + '\n')
    for rt in R:
        for d, v in rep['routes'][rt]['deltas'].items():
            print(rt, 'delta', d, 'sel_err', round(v['selection_error'], 3), json.dumps(v['all']), 'weighted', json.dumps({k: round(x, 3) for k, x in v['natural_weighted'].items()}))
    print('TAU', tau, json.dumps(cdev[tau]))
    for d, v in ab['eval'].items():
        print('eval delta', d, json.dumps({rt: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in vv['at_tau'].items()} for rt, vv in v.items()}))
    if route == 'both': print('COMPARISON', json.dumps(rep['comparison'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
