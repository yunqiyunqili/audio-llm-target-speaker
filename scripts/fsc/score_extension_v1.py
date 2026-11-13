"""FSC extension v1 one-shot scoring (docs/fsc_extension_plan_v1.md; mapping configs/fsc_mapping_v1.json). CPU only. FROZEN.

Prediction -> FSC triple: parse the model output (slu_normalize.parse), then the frozen mapping (object / location keywords
matched as whole words in normalize_v2 + rescore_slu_v2.toks of each entity value, in output order; SLURP intent -> action and
default object; object-implied actions). The transcript field is never used.
Metrics: target_any (PRIMARY; >= 1 non-'none' gold slot predicted equal), intent_exact, intent_family (action+object), per-slot
accuracy, unmapped-action and none-object coverage, parse failures. Two-speaker: binding_score_v1.judge on (slot=value) multisets.
Primary comparisons (Audio LoRA - Whisper+T2-LoRA, target_any, paired; Bonferroni 3 -> 98.33% and 95%):
  clean, reverb : bootstrap grouped by transcription text (as SLURP slurp_id; compare_audio_vs_t2.boot rule), B=2000, rng 2026;
  spatial       : selected stream (C5), equal weights, judge(...)['t_any']; group-level bootstrap stratified by angle stratum
                  (eval_final_lora_checks construction). Significant <=> 98.33% lower bound > 0.
Descriptive: both LoRAs vs their own baselines, C0 (gold transcription), c6-real-v1 main conditions (8-sector centre, sigma10 x10,
abstention A margin >= 17 deg, B sector) with cv_dev93 natural weights, interferer risk by actual route, text-overlap
sensitivity, pilot-64 exploratory rescoring (--pilot).
Inputs JSON: {"list": configs/fsc_ext_v1_list.json, "selection_dir": ..., "pred": {cond: {route: [predictions.jsonl, ...]}}}
  conds: clean, reverb, c0, spatial_selected, spatial_other; routes: base, audio_lora, wt2, t2_lora (c0: t2, t2_lora).
Prediction records (scripts/fsc/ext_batch.py): item_id, condition, route, raw_output (+ asr for Whisper routes).
Usage: python -I -B scripts/fsc/score_extension_v1.py --inputs INPUTS.json --out DIR
       python -I -B scripts/fsc/score_extension_v1.py --pilot --out DIR      (exploratory, pilot v2 predictions)
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for d in ('scripts/qwen', 'scripts/slurp'):
    sys.path.insert(0, str(ROOT / d))
from binding_score_v1 import judge  # noqa: E402
from rescore_slu_v2 import normalize_v2, toks  # noqa: E402
from slu_normalize import parse  # noqa: E402
import eval_c6_real as real  # noqa: E402  (sector, choose, wrap, TAU_A, NATURAL; unchanged)

MAP = json.load(open(ROOT / 'configs/fsc_mapping_v1.json'))
SLOTS = ('action', 'object', 'location')
OBJ = [(o, set(v['words'])) for o, v in MAP['object_keywords'].items() if isinstance(v, dict) and v.get('words')]
LOC = [(o, set(v['words'])) for o, v in MAP['location_keywords'].items() if isinstance(v, dict) and 'words' in v]
I2A = MAP['intent_to_action']
IMPL = {o: a for a in ('bring', 'change language') for o in MAP['object_implies_action'][a]}
LANGW = set(MAP['object_implies_action']['language_words']['words'])
Q9833 = [0.05 / 3 / 2, 1 - 0.05 / 3 / 2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predict(raw):
    """raw model output -> (triple, info). Frozen mapping, structured output only."""
    obj, st = parse(raw)
    if not obj:
        return dict(action='unmapped', object='none', location='none'), dict(parse_ok=False, intent=None)
    vals = [toks(normalize_v2(v)) for _, v in obj['entities']]
    o = None
    for ws in vals:
        hit = [name for name, kw in OBJ if ws & kw]
        if hit:
            o = 'lamp' if ('lamp' in hit and 'lights' in hit) else hit[0]; break
    loc = next((name for ws in vals for name, kw in LOC if ws & kw), 'none')
    m = I2A.get(obj['intent'], I2A['*'])
    if o is None:
        o = m['default_object'] or 'none'
    a = m['action']
    if o in IMPL:
        a = IMPL[o]
    elif any(ws & LANGW for ws in vals):
        a = 'change language'
        if o not in IMPL:
            o = 'none'
    return dict(action=a, object=o, location=loc), dict(parse_ok=st == 'ok', intent=obj['intent'])


def single_row(raw, gold):
    p, info = predict(raw)
    G = {s: gold[s] for s in SLOTS}
    nz = [s for s in SLOTS if G[s] != 'none']
    return dict(target_any=any(p[s] == G[s] for s in nz), exact=all(p[s] == G[s] for s in SLOTS),
                family=p['action'] == G['action'] and p['object'] == G['object'], slot={s: p[s] == G[s] for s in SLOTS},
                unmapped=p['action'] == 'unmapped', none_obj=p['object'] == 'none', parse_ok=info['parse_ok'], pred=p)


def sv(t):
    return collections.Counter(f'{s}={t[s]}' for s in SLOTS if t[s] != 'none')


def mix_row(raw, tgt, itf):
    p, info = predict(raw)
    VT, VI, H = sv(tgt), sv(itf), sv(p)
    j = judge(VT - VI, VI - VT, H)
    return dict(cat=j['cat'], t_any=j['t_any'], t_all=j['t_all'], exact=all(p[s] == tgt[s] for s in SLOTS), parse_ok=info['parse_ok'],
                unmapped=p['action'] == 'unmapped', dt=sum(((VT - VI) & H).values()), di=sum(((VI - VT) & H).values()), npred=sum(H.values()))


def load(paths):
    R = {}
    for p in paths:
        for line in open(ROOT / p):
            r = json.loads(line); assert r['item_id'] not in R, r['item_id']; R[r['item_id']] = r
    return R


def boot_text(d):
    """compare_audio_vs_t2.boot rule (groups in insertion order), + 98.33% from the same draws."""
    by = collections.defaultdict(list)
    for k, v in d.items():
        by[v[0]].append(v[1])
    s = np.array([sum(v) for v in by.values()]); c = np.array([len(v) for v in by.values()])
    ix = np.random.default_rng(2026).integers(0, len(s), (2000, len(s))); z = s[ix].sum(1) / c[ix].sum(1)
    return dict(delta=float(s.sum() / c.sum()), ci95=[float(x) for x in np.quantile(z, [.025, .975])], ci9833=[float(x) for x in np.quantile(z, Q9833)],
                n=int(c.sum()), n_text=len(s), B=2000, seed=2026)


def summ_single(rows):
    n = len(rows)
    return dict(n=n, target_any=sum(r['target_any'] for r in rows) / n, intent_exact=sum(r['exact'] for r in rows) / n,
                intent_family=sum(r['family'] for r in rows) / n, **{f'slot_{s}': sum(r['slot'][s] for r in rows) / n for s in SLOTS},
                action_unmapped=sum(r['unmapped'] for r in rows) / n, object_none=sum(r['none_obj'] for r in rows) / n,
                parse_failures=sum(not r['parse_ok'] for r in rows))


def score_single(items, P, routes, pairs):
    out = dict(routes={}, paired={})
    rows = {rt: {k: single_row(P[rt][k]['raw_output'], items[k]['target']) for k in items} for rt in routes}
    for rt in routes:
        out['routes'][rt] = summ_single(list(rows[rt].values()))
        out['routes'][rt]['overlap_excluded_descriptive'] = summ_single([rows[rt][k] for k in items if not items[k]['slurp_train_overlap']])
    for a, b in pairs:
        if a in rows and b in rows:
            out['paired'][f'{a}-{b}'] = boot_text({k: (items[k]['text'], int(rows[a][k]['target_any']) - int(rows[b][k]['target_any'])) for k in items})
            out['paired'][f'{a}-{b}_overlap_excluded_descriptive'] = boot_text({k: (items[k]['text'], int(rows[a][k]['target_any']) - int(rows[b][k]['target_any']))
                                                                                 for k in items if not items[k]['slurp_train_overlap']})
    return out, rows


def score_spatial(L, sel, ref, P, routes):
    """C5 + c6-real-v1 main conditions on FSC triples; mirrors eval_final_lora_checks.py (queries, abstention, weights, bootstrap)."""
    G = {g['group']: g for g in L['groups']}
    keys = sorted(f"g{g['group']:03d}_s{sw['swap']}" for g in L['groups'] for sw in g['swaps']); n = len(keys)
    gid = np.array([int(k[1:4]) for k in keys]); swp = {f"g{g['group']:03d}_s{sw['swap']}": sw for g in L['groups'] for sw in g['swaps']}
    th1 = np.array([G[g]['rir']['theta1'] for g in gid]); th2 = np.array([G[g]['rir']['theta2'] for g in gid])
    strata = [G[g]['rir']['stratum'] for g in gid]
    W = np.array([real.NATURAL[s] / sum(real.NATURAL.values()) / strata.count(s) for s in strata]); E = np.ones(n)
    az = np.array([sel[k]['doa'] for k in keys]); s0 = np.array([sel[k]['stream'] for k in keys]); refs = np.array([ref[k]['ref_stream'] for k in keys])
    idx = {g: np.flatnonzero(gid == g) for g in sorted(G)}
    by = [sorted(set(gid[np.array([s == st for s in strata])])) for st in real.STRATA]
    rng = np.random.default_rng(2026)
    BOOT = np.array([np.concatenate([idx[g] for gs in by for g in rng.choice(gs, size=len(gs), replace=True)]) for _ in range(2000)])
    T = {}
    for rt in routes:
        T[rt] = []
        for st in ('selected', 'other'):
            res = {}
            for k in keys:
                g = G[int(k[1:4])]; t = swp[k]['target']; i = 'B' if t == 'A' else 'A'
                res[k] = mix_row(P[st][rt][k]['raw_output'], g[t]['target'], g[i]['target'])
            T[rt].append(res)
    Qs = {'C5': (th1[None], np.zeros((1, n), bool)),
          'sector_K8_center': (np.array([[real.sector(t, 8, 'center')[1] for t in th1]]),
                               np.array([[real.sector(a, 8, 'center')[0] == real.sector(b, 8, 'center')[0] for a, b in zip(th1, th2)]]))}
    NG = len(G)
    Qs['noise_sigma10'] = (np.array([[real.wrap(t + e[g]) for t, g in zip(th1, gid)] for e in
                                     [np.random.default_rng(np.random.SeedSequence([2026, 10, r])).normal(0, 10, NG) for r in range(10)]]), np.zeros((10, n), bool))

    def rate(x, m, w):
        den = (m * w).sum(1)
        return float(np.mean((x * m * w).sum(1) / den)) if np.all(den > 0) else None

    def ci(x, m, w, q):
        den = (m[:, BOOT] * w[BOOT]).sum(2); num = (x[:, BOOT] * m[:, BOOT] * w[BOOT]).sum(2)
        z = np.nanmean(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0), axis=0)
        return np.nanquantile(z, q).tolist()
    rep = {}
    for cn, (Q, ind) in Qs.items():
        ch = np.array([[real.choose(az[i], q) for i, q in enumerate(qr)] for qr in Q])
        if cn == 'C5':
            assert np.array_equal(ch[0], s0)
        mg = np.array([[abs(real.wrap(q - az[i][1 - ch[r, i]])) - abs(real.wrap(q - az[i][ch[r, i]])) for i, q in enumerate(qr)] for r, qr in enumerate(Q)])
        wrong = ch != refs[None]
        pol = {'none': np.ones_like(ch, bool)}
        if cn != 'C5':
            pol['A'] = mg >= real.TAU_A
        if cn.startswith('sector'):
            pol['B'] = np.array([[not (real.sector(az[i][0], 8, 'center')[0] == real.sector(q, 8, 'center')[0] and real.sector(az[i][1], 8, 'center')[0] == real.sector(q, 8, 'center')[0])
                                  for i, q in enumerate(Q[0])]])
        A = {}
        for rt in routes:
            rows = [[T[rt][0 if ch[r, i] == s0[i] else 1][k] for i, k in enumerate(keys)] for r in range(len(Q))]
            arr = {f: np.array([[x[f] for x in rr] for rr in rows]) for f in ('t_any', 'cat', 'dt', 'di', 'npred', 'exact')}
            A[rt] = dict(target_any=(arr['t_any'] & ~ind).astype(float), only_interferer=(arr['cat'] == 'only_interferer').astype(float),
                         only_target=((arr['cat'] == 'only_target') & ~ind).astype(float), exact=(arr['exact'] & ~ind).astype(float), raw=arr)
        res = {}
        for p, ans in pol.items():
            d = dict(coverage_eq=rate(ans.astype(float), np.ones_like(ans), E), coverage_nat=rate(ans.astype(float), np.ones_like(ans), W),
                     sel_error_main_eq=rate((wrong | ind).astype(float), ans, E), sel_error_actual_eq=rate(wrong.astype(float), ans, E), models={}, paired={})
            for rt in routes:
                d['models'][rt] = {f'{f}_{lab}': rate(A[rt][f], ans, w) for f in ('target_any', 'only_interferer', 'only_target', 'exact') for lab, w in (('eq', E), ('nat', W))}
                if p == 'none':
                    d['models'][rt]['by_actual_route'] = {lab: dict(n_observations=int(m.sum()), target_value_hits=int(A[rt]['raw']['dt'][m].sum()), interferer_value_hits=int(A[rt]['raw']['di'][m].sum()),
                                                                     only_interferer=int(((A[rt]['raw']['cat'] == 'only_interferer') & m).sum()))
                                                          for lab, m in (('correct', ~wrong), ('wrong', wrong))}
                    if cn == 'C5':
                        otc = A[rt]['raw']['cat'][0] == 'only_target'
                        d['models'][rt]['pair_flip_accuracy'] = float(np.mean([otc[idx[g]].all() for g in sorted(G)]))
            for a, b in (('audio_lora', 't2_lora'), ('audio_lora', 'base'), ('t2_lora', 'wt2')):
                if a in A and b in A:
                    for f in ('target_any', 'only_interferer'):
                        x = A[a][f] - A[b][f]
                        d['paired'][f'{a}-{b}/{f}'] = {lab: dict(delta=rate(x, ans, w), ci95=ci(x, ans, w, [.025, .975]), ci9833=ci(x, ans, w, Q9833)) for lab, w in (('eq', E), ('nat', W))}
            res[p] = d
        rep[cn] = res
    return rep


def main(inputs, out):
    out = Path(out); assert not out.exists()
    I = json.load(open(inputs)); L = json.load(open(ROOT / I['list']))
    items = {it['item_id']: it for it in L['single']}
    rep = dict(spec='docs/fsc_extension_plan_v1.md', mapping_sha256=sha(ROOT / 'configs/fsc_mapping_v1.json'), list_sha256=sha(ROOT / I['list']),
               inputs=str(inputs), inputs_sha256=sha(inputs), primary={}, conditions={})
    R4 = ('base', 'audio_lora', 'wt2', 't2_lora'); PAIRS = (('audio_lora', 't2_lora'), ('audio_lora', 'base'), ('t2_lora', 'wt2'))
    for cond in ('clean', 'reverb'):
        P = {rt: load(I['pred'][cond][rt]) for rt in I['pred'][cond]}
        assert all(set(v) == set(items) for v in P.values()), cond
        assert all(P['wt2'][k].get('asr') == P['t2_lora'][k].get('asr') for k in items), f'{cond}: wt2 and t2_lora must share the Whisper transcript'
        rep['conditions'][cond], _ = score_single(items, P, [r for r in R4 if r in P], PAIRS)
        rep['primary'][cond] = dict(rep['conditions'][cond]['paired']['audio_lora-t2_lora'], a=rep['conditions'][cond]['routes']['audio_lora']['target_any'],
                                    b=rep['conditions'][cond]['routes']['t2_lora']['target_any'])
    if 'c0' in I['pred']:
        P = {rt: load(I['pred']['c0'][rt]) for rt in I['pred']['c0']}
        rep['conditions']['c0'], _ = score_single(items, P, list(P), (('t2_lora', 't2'),))
    sd = ROOT / I['selection_dir']
    sel = {r['key']: r for r in map(json.loads, open(sd / 'selection.jsonl'))}; ref = {r['key']: r for r in map(json.loads, open(sd / 'per_sample.jsonl'))}
    P = {st: {rt: load(I['pred'][f'spatial_{st}'][rt]) for rt in I['pred'][f'spatial_{st}']} for st in ('selected', 'other')}
    routes = [r for r in R4 if r in P['selected']]
    sp = score_spatial(L, sel, ref, P, routes); rep['conditions']['spatial'] = sp
    c5 = sp['C5']['none']; pa = c5['paired']['audio_lora-t2_lora/target_any']['eq']
    rep['primary']['spatial'] = dict(pa, a=c5['models']['audio_lora']['target_any_eq'], b=c5['models']['t2_lora']['target_any_eq'], n=len(sel),
                                     natural_weighted_descriptive=c5['paired']['audio_lora-t2_lora/target_any']['nat'])
    for c, p in rep['primary'].items():
        p['significant_bonferroni'] = p['ci9833'][0] > 0; p['lower95_gt_0'] = p['ci95'][0] > 0
    out.mkdir(parents=True)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for c, p in rep['primary'].items():
        print('PRIMARY', c, f"audio {p['a']:.4f} t2lora {p['b']:.4f} delta {p['delta']:+.4f} 95% [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
              f"98.33% [{p['ci9833'][0]:+.4f},{p['ci9833'][1]:+.4f}] significant={p['significant_bonferroni']}", flush=True)
    print('REPORT', out / 'report.json', sha(out / 'report.json'))


def pilot(out):
    """Exploratory: pilot v2 predictions (64 x 4 conditions x 3 routes) rescored with the frozen mapping. Single conditions only
    plus spatial / direction selected-stream target_any against the gold triple (no c6-real; pilot has no other-stream runs)."""
    out = Path(out); assert not out.exists()
    O = ROOT / 'outputs/fsc_pilot'; T = [json.loads(x) for x in open(O / 'pilot_tasks.jsonl')]
    dirs = dict(base='pred_v2_base_22076114', audio_lora='pred_v2_audio_lora_22076115', t2_lora='pred_v2_t2_lora_22076116')
    rep = dict(note='EXPLORATORY ONLY; pilot 64 utterances; not part of the extension results', mapping_sha256=sha(ROOT / 'configs/fsc_mapping_v1.json'), conditions={})
    for cond in ('clean', 'reverb', 'spatial', 'direction_uncertainty'):
        rows = {}
        for rt, d in dirs.items():
            Pp = [json.loads(x) for x in open(O / d / 'predictions.jsonl')]
            rows[rt] = {i: single_row(p['raw_output'], t['target']) for i, (p, t) in enumerate(zip(Pp, T)) if t['condition'] == cond}
        rep['conditions'][cond] = dict(routes={rt: summ_single(list(v.values())) for rt, v in rows.items()},
                                       audio_minus_t2lora=boot_text({i: (T[i]['transcript'], int(rows['audio_lora'][i]['target_any']) - int(rows['t2_lora'][i]['target_any'])) for i in rows['base']}))
    out.mkdir(parents=True); (out / 'pilot_exploratory.json').write_text(json.dumps(rep, indent=1) + '\n')
    for c, v in rep['conditions'].items():
        print(c, {rt: (round(s['target_any'], 3), round(s['intent_exact'], 3)) for rt, s in v['routes'].items()}, round(v['audio_minus_t2lora']['delta'], 3), [round(x, 3) for x in v['audio_minus_t2lora']['ci95']])
    print('SHA256', sha(out / 'pilot_exploratory.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs'); ap.add_argument('--out', required=True); ap.add_argument('--pilot', action='store_true')
    a = ap.parse_args()
    pilot(a.out) if a.pilot else main(a.inputs, a.out)
