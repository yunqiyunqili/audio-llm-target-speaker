"""Real C6 v1 evaluation (docs/c6_real_v1.md, frozen 2026-11-06 11:05 EST). CPU only; reads existing raw outputs only.
User-provided query direction (C6-sector: sector centre; C6-noise: theta1 + N(0, sigma^2), per group, 10 repeats); system-estimated
stream directions (stored v3 DOAs); v3 choose(). Main scoring: all 200 samples, direction-indistinguishable samples count as
selection errors and get no target credit. Abstention A (margin < 17 deg) and B (both estimated DOAs inside the query sector,
C6-sector only). Equal-angle and natural-angle-weighted metrics; stratified group bootstrap 95% CI (B=2000, seed 2026).
SpatialNet = IN-TRAINING snapshot (ep72), diagnosis only.
Usage: python -I -B scripts/qwen/eval_c6_real.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402
from eval_c6_delta import NATURAL, TAUS, choose, load, wrap  # noqa: E402
import split_cfg  # noqa: E402  (2026-11-12 SLURP test switch: list/manifest/group count; devel default unchanged)

O = PROJ / 'outputs'
MANIFEST = split_cfg.MANIFEST  # devel: <SHARED_ROOT>/data/mix_v1/devel_seed2026/manifest.jsonl
TAU_A = 17
SIGMAS = (5, 10, 15)
REPS = 10
SECTORS = [(8, 'center'), (4, 'center'), (8, 'rotated'), (4, 'rotated')]
NBOOT = 2000
STRATA = list(NATURAL)
CMP = ('only_target', 'only_interferer', 'both', 'neither', 'target_any', 'target_all', 'pair_flip_accuracy', 'M', 'U')
FE = dict(
    spatialnet=dict(sel=O / 'frontend_snapshot_eval/spatialnet/selection.jsonl', ref=('snap', O / 'frontend_snapshot_eval/spatialnet/per_sample.jsonl'),
                    routes=dict(P2=(O / 'qwen_omni_snapshot/22016930_spatialnet/P2/raw.jsonl', O / 'qwen_omni_snapshot_other/22017085_spatialnet/P2/raw.jsonl'),
                                WT2=(O / 'qwen_omni_snapshot/22016930_spatialnet/T2/raw.jsonl', O / 'qwen_omni_snapshot_other/22017085_spatialnet/T2/raw.jsonl'))),
    fasnet_tac=dict(sel=O / 'c5_v1/selection.jsonl', ref=('v2', O / 'frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl'),
                    routes=dict(P2=(O / 'qwen_omni_c5/22015337/C5_fasnet_tac/raw.jsonl', O / 'qwen_omni_c6c7/22015445/OTHER/raw.jsonl'))))
STORED = dict(fasnet_tac=dict(P2=lambda: json.load(open(O / 'qwen_omni_c5_eval/22015337/eval_c5_v1.json'))['conditions']['C5_fasnet_tac']['all']),
              spatialnet=dict(P2=lambda: json.load(open(O / 'frontend_snapshot_downstream_eval/eval.json'))['models']['spatialnet']['P2'],
                              WT2=lambda: json.load(open(O / 'frontend_snapshot_downstream_eval/eval.json'))['models']['spatialnet']['WhisperT2']))


def sector(th, K, scheme):
    W = 360.0 / K
    if scheme == 'center':
        k = int(((th + W / 2) % 360.0) // W); return k, wrap(k * W)
    k = int((th % 360.0) // W); return k, wrap((k + 0.5) * W)


def main(out, *, final_note=None, spatial_delta_reference=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(split_cfg.LIST))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(MANIFEST))}
    G = len(L); assert sorted(L) == list(range(G))
    keys = sorted(man); n = len(keys); assert n == 2 * G and (split_cfg.SPLIT != 'devel' or n == 200)
    grp = np.array([int(k[1:4]) for k in keys]); th1 = np.array([L[g]['rir']['theta1'] for g in grp]); th2 = np.array([L[g]['rir']['theta2'] for g in grp])
    strat = np.array([STRATA.index(L[g]['rir']['stratum']) for g in grp])
    ns = np.bincount(strat, minlength=4); Wn = sum(NATURAL.values())
    vnat = np.array([NATURAL[STRATA[s]] / Wn / ns[s] for s in strat])
    # stratified group bootstrap indices (shared by every condition)
    rng = np.random.default_rng(2026)
    gidx = {g: [i for i in range(n) if grp[i] == g] for g in range(G)}
    gby = [sorted({int(grp[i]) for i in range(n) if strat[i] == s}) for s in range(4)]
    BOOT = np.array([np.concatenate([gidx[g] for s in range(4) for g in rng.choice(gby[s], size=len(gby[s]), replace=True)]) for _ in range(NBOOT)])
    assert all((np.bincount(strat[b], minlength=4) == ns).all() for b in BOOT[:5])

    # query directions per condition: array (R, n) of theta_query, sector info
    conds = {}
    for K, sch in SECTORS:
        q = np.array([[sector(t, K, sch)[1] for t in th1]])
        kq = np.array([[sector(t, K, sch)[0] for t in th1]])
        ind = np.array([[sector(a, K, sch)[0] == sector(b, K, sch)[0] for a, b in zip(th1, th2)]])
        conds[f'sector_K{K}_{sch}'] = dict(kind='sector', K=K, scheme=sch, q=q, kq=kq, indist=ind)
    for sg in SIGMAS:
        Q = []
        for r in range(REPS):
            eps = np.random.default_rng(np.random.SeedSequence([2026, sg, r])).normal(0, sg, G)
            Q.append([wrap(t + eps[g]) for t, g in zip(th1, grp)])
        conds[f'noise_sigma{sg}'] = dict(kind='noise', sigma=sg, q=np.array(Q), indist=np.zeros((REPS, n), bool))
    ind8 = conds['sector_K8_center']['indist'][0]

    def rate(x, m, w, idx=None):
        """x, m: (R, n); w: (n,). Returns per-repeat (R,) weighted mean of x over mask m (nan if empty)."""
        if idx is not None:
            x, m, w = x[:, idx], m[:, idx], w[idx]
        den = (m * w).sum(1)
        return np.where(den > 0, (x * m * w).sum(1) / np.where(den > 0, den, 1), np.nan)

    def _m(v):
        return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))

    def stat(arr, mask, cover_base, ci=True):
        """arr: dict of (R, n) metric arrays; mask: answered (R, n). Point = mean over repeats; range; bootstrap CI of the repeat mean."""
        res = {}
        ones = np.ones(n)
        for wn, w in (('eq', ones), ('nat', vnat)):
            fields = dict(coverage=(mask.astype(float), cover_base), **{m: (a, mask) for m, a in arr.items()})
            for m, (x, mk) in fields.items():
                pr = rate(x, mk, w); d = dict(mean=None if np.all(np.isnan(pr)) else float(np.nanmean(pr)))
                if pr.shape[0] > 1:
                    d.update(min=float(np.nanmin(pr)), max=float(np.nanmax(pr)))
                if ci and m in ('coverage', 'sel_error', 'only_interferer', 'target_any'):
                    bs = np.array([_m(rate(x, mk, w, b)) for b in BOOT])
                    bs = bs[~np.isnan(bs)]
                    d['ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else None
                res[f'{m}_{wn}'] = d
        res['answered_mean'] = float(mask.sum(1).mean())
        return res

    rep = dict(spec='docs/c6_real_v1.md', note='SpatialNet = ep72 IN-TRAINING snapshot (快照诊断); FaSNet-TAC = control (P2 only); no GPU, no test; '
               'all thresholds empirical', tau_A=TAU_A, n_boot=NBOOT, natural_weights=NATURAL, frontends={}, consistency={})
    rc_plot = {}
    if final_note is not None:
        rep['note'] = final_note
    for fe, F in FE.items():
        sel = {r['key']: r for r in map(json.loads, open(F['sel']))}
        if F['ref'][0] == 'snap':
            ref = {r['key']: r['ref_stream'] for r in map(json.loads, open(F['ref'][1]))}
        else:
            ref = {r['key']: r['ref_stream_direct'] for r in map(json.loads, open(F['ref'][1])) if r['frontend'] == 'fasnet_tac'}
        az = np.array([sel[k]['doa'] for k in keys]); s0 = np.array([sel[k]['stream'] for k in keys]); refa = np.array([ref[k] for k in keys])
        R = {rt: (load(a), load(b)) for rt, (a, b) in F['routes'].items()}
        for rt, (RS, RO) in R.items():
            for i, k in enumerate(keys):
                assert RS[k]['file'] == k and RO[k]['file'] == k
                if 'stream' in RS[k]:
                    assert RS[k]['stream'] == s0[i] and RO[k]['stream'] == 1 - s0[i]
        # ---- consistency: query = theta1 ----
        c_or = np.array([choose(az[i], th1[i]) for i in range(n)])
        assert (c_or == s0).all(), f'{fe}: selection mismatch at theta1'
        rep['consistency'][fe] = {}
        for rt, (RS, RO) in R.items():
            got = summ(score({k: (RS[k] if c_or[i] == s0[i] else RO[k]) for i, k in enumerate(keys)}, L, man)); exp = STORED[fe][rt]()
            bad = {m: (got[m], exp[m]) for m in CMP if abs(got[m] - exp[m]) > 1e-12}
            assert not bad, f'{fe}/{rt}: consistency FAILED {bad}'
            rep['consistency'][fe][rt] = dict(passed=True, **{m: got[m] for m in CMP})
        print('CONSISTENCY_PASS', fe, flush=True)
        # category arrays for both streams per route: cat[rt][stream_is_selected] per sample
        cats = {}
        for rt, (RS, RO) in R.items():
            rs = {r['key']: r for r in score(RS, L, man)}; ro = {r['key']: r for r in score(RO, L, man)}
            cats[rt] = (rs, ro)
        res_fe = dict(conditions={}); rc_plot[fe] = {}
        for cn, C in conds.items():
            Q = C['q']; Rn = Q.shape[0]
            ch = np.array([[choose(az[i], Q[r, i]) for i in range(n)] for r in range(Rn)])
            mg = np.array([[abs(wrap(Q[r, i] - az[i][1 - ch[r, i]])) - abs(wrap(Q[r, i] - az[i][ch[r, i]])) for i in range(n)] for r in range(Rn)])
            ind = C['indist']; wrong_act = ch != refa[None]; sw = wrong_act | ind
            ansA = mg >= TAU_A
            ansB = None
            if C['kind'] == 'sector':
                K, sch = C['K'], C['scheme']
                kd = np.array([[sector(az[i][j], K, sch)[0] for j in range(2)] for i in range(n)])
                ansB = ~((kd[:, 0] == C['kq'][0]) & (kd[:, 1] == C['kq'][0]))[None]
            full = np.ones((Rn, n), bool)
            cres = dict(kind=C['kind'], indist_count=int(ind[0].sum()) if C['kind'] == 'sector' else 0,
                        indist_by_stratum={STRATA[s]: int(ind[0][strat == s].sum()) for s in range(4)} if C['kind'] == 'sector' else None,
                        indist_nat_weighted=float((vnat * ind[0]).sum()) if C['kind'] == 'sector' else 0.0,
                        selection_error_actual_vs_ref=float(wrong_act.mean()), routes={})
            if ansB is not None:
                cres['B_flag_vs_indist'] = dict(abstainB_and_indist=int((~ansB[0] & ind[0]).sum()), abstainB_and_dist=int((~ansB[0] & ~ind[0]).sum()),
                                                answerB_and_indist=int((ansB[0] & ind[0]).sum()), answerB_and_dist=int((ansB[0] & ~ind[0]).sum()))
            for rt in R:
                rs, ro = cats[rt]
                row = [[(rs if ch[r, i] == s0[i] else ro)[keys[i]] for i in range(n)] for r in range(Rn)]
                cat = np.array([[x['cat'] for x in rr] for rr in row])
                ta = np.array([[x['t_any'] for x in rr] for rr in row]); tal = np.array([[x['t_all'] for x in rr] for rr in row])
                ot = cat == 'only_target'; oi = cat == 'only_interferer'
                main_arr = dict(sel_error=sw.astype(float), only_interferer=oi.astype(float), only_target=(ot & ~ind).astype(float),
                                target_any=(ta & ~ind).astype(float), target_all=(tal & ~ind).astype(float), neither=(cat == 'neither').astype(float))
                act_arr = dict(sel_error=wrong_act.astype(float), only_interferer=oi.astype(float), only_target=ot.astype(float),
                               target_any=ta.astype(float), target_all=tal.astype(float), neither=(cat == 'neither').astype(float))
                # pair flip (main scoring, no abstention): both swaps only_target with credit
                otc = ot & ~ind
                pf = [np.mean([otc[r][gidx[g]].all() for g in range(G)]) for r in range(Rn)]
                rr = dict(main=dict(none=stat(main_arr, full, full), A=stat(main_arr, ansA, full)),
                          pair_flip_main=dict(mean=float(np.mean(pf)), min=float(np.min(pf)), max=float(np.max(pf))),
                          strata_eq={})
                if ansB is not None:
                    rr['main']['B'] = stat(main_arr, ansB, full)
                    dist = ~ind
                    rr['distinguishable'] = dict(none=stat(act_arr, dist, full, ci=False), A=stat(act_arr, dist & ansA, dist, ci=False),
                                                 B=stat(act_arr, dist & ansB, dist, ci=False))
                    rr['indistinguishable_actual'] = {m: float(act_arr[m][0][ind[0]].mean()) if ind[0].any() else None for m in act_arr}
                else:
                    rr['by_true_K8_center_label'] = dict(indist=stat(act_arr, np.broadcast_to(ind8, (Rn, n)), full, ci=False),
                                                         dist=stat(act_arr, np.broadcast_to(~ind8, (Rn, n)), full, ci=False))
                for s in range(4):
                    ms = np.broadcast_to(strat == s, (Rn, n))
                    d = dict(none={m: float(np.nanmean(rate(a, ms, np.ones(n)))) for m, a in main_arr.items()},
                             A_coverage=float(np.nanmean(rate(ansA.astype(float), ms, np.ones(n)))),
                             A_sel_error=float(np.nanmean(rate(sw.astype(float), ms & ansA, np.ones(n)))) if (ms & ansA).any() else None)
                    if ansB is not None:
                        d.update(B_coverage=float(rate(ansB.astype(float), ms, np.ones(n))[0]),
                                 B_sel_error=float(rate(sw.astype(float), ms & ansB, np.ones(n))[0]) if (ms & ansB).any() else None)
                    rr['strata_eq'][STRATA[s]] = d
                cres['routes'][rt] = rr
            # risk-coverage (equal weights, main sel error), tau 0..90
            rc_plot[fe][cn] = dict(curve=[dict(tau=t, coverage=float(np.mean((mg >= t).mean(1))),
                                               sel_error=_m(rate(sw.astype(float), mg >= t, np.ones(n))))
                                          for t in TAUS],
                                   B=(dict(coverage=float(ansB.mean()), sel_error=float(rate(sw.astype(float), ansB, np.ones(n))[0])) if ansB is not None else None))
            res_fe['conditions'][cn] = cres
            print('DONE', fe, cn, flush=True)
        rep['frontends'][fe] = res_fe

    # ---- mainline judgment (frozen rule, SpatialNet + P2, natural-weighted point estimates) ----
    judg = {}
    for cn, rules in (('sector_K8_center', ('A', 'B')), ('noise_sigma10', ('A',))):
        M = rep['frontends']['spatialnet']['conditions'][cn]['routes']['P2']['main']
        oi0 = M['none']['only_interferer_nat']['mean']
        eff = {}
        for x in rules:
            oix = M[x]['only_interferer_nat']['mean']; cov = M[x]['coverage_nat']['mean']
            eff[x] = dict(only_interferer_nat=oix, coverage_nat=cov, effective=bool(oix is not None and oix <= 0.5 * oi0 + 1e-12 and cov >= 0.60 - 1e-12))
        if oi0 >= 0.05 - 1e-12 and any(v['effective'] for v in eff.values()):
            v = '"方向不确定下减少答错"是可行主线'
        elif oi0 < 0.02:
            v = '方向不确定性在该条件下是次要问题'
        else:
            v = '只在小夹角或部分条件下重要'
        judg[cn] = dict(OI0_nat=oi0, rules=eff, verdict=v)
    rep['mainline_judgment'] = judg
    # C6-delta reference (existing results, not recomputed); devel only (C6-delta is not run on the SLURP test)
    if split_cfg.SPLIT != 'devel':
        rep['c6delta_reference'] = None
        (out / 'eval_c6_real.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
        (out / 'risk_coverage.json').write_text(json.dumps(dict(tau_A=TAU_A, curves=rc_plot), indent=1) + '\n')
        print('MAINLINE', json.dumps(judg, ensure_ascii=False, indent=1)); return
    fd = json.load(open(O / 'c6_delta_v1/eval_c6_delta.json')); sd = json.load(open(spatial_delta_reference or O / 'c6_delta_spatialnet_v1/eval_c6_delta_spatialnet.json'))
    rep['c6delta_reference'] = {d: dict(fasnet_P2=dict(sel_error=fd['deltas'][d]['selection_error'], oi=fd['deltas'][d]['all']['only_interferer'], ta=fd['deltas'][d]['all']['target_any'],
                                                       oi_nat=fd['deltas'][d]['natural_weighted']['only_interferer'], ta_nat=fd['deltas'][d]['natural_weighted']['target_any']),
                                        spatialnet_P2=dict(sel_error=sd['routes']['P2']['deltas'][d]['selection_error'], oi=sd['routes']['P2']['deltas'][d]['all']['only_interferer'],
                                                           ta=sd['routes']['P2']['deltas'][d]['all']['target_any'], oi_nat=sd['routes']['P2']['deltas'][d]['natural_weighted']['only_interferer'],
                                                           ta_nat=sd['routes']['P2']['deltas'][d]['natural_weighted']['target_any'])) for d in fd['deltas']}
    (out / 'eval_c6_real.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    (out / 'risk_coverage.json').write_text(json.dumps(dict(tau_A=TAU_A, curves=rc_plot), indent=1) + '\n')
    print('MAINLINE', json.dumps(judg, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
