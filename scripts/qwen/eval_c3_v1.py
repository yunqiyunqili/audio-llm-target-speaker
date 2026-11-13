"""Evaluate C1 / C3 (P2 prompt) per docs/c3_gate_v1.md (frozen). Offline; reads raw.jsonl; writes a NEW dir.
Frozen code imported unchanged: rescore_slu_v2 (score_item, f1, INV), binding_score_v1 (vals, judge),
postproc_c1_ext (REFUSAL). Gate: c3-gate-v1 (group bootstrap, 10,000 resamples, numpy default_rng(2026)).
Usage: python -I scripts/qwen/eval_c3_v1.py RUN_DIR MIX_MANIFEST P2_RUN_DIR OUT_DIR
  RUN_DIR: outputs/qwen_omni_c3_p2/<job>; MIX_MANIFEST: data/mix_v1/devel_seed2026/manifest.jsonl;
  P2_RUN_DIR: outputs/qwen_omni_c1_p2t2/22011390/P2 (for the 50-recording consistency check)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from rescore_slu_v2 import INV, f1, normalize_v2, score_item  # noqa: E402
from binding_score_v1 import judge, vals  # noqa: E402
from postproc_c1_ext import REFUSAL  # noqa: E402

CATS = ('only_target', 'only_interferer', 'both', 'neither')
N_BOOT, SEED = 10000, 2026
BINS = dict(stratum=None, t60=[(0.2, 0.3), (0.3, 0.4), (0.4, 0.5000001)],
            overlap=[(0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0000001)],
            sir_full=[(-np.inf, -0.5), (-0.5, 0.5000001), (0.5000001, np.inf)])


def slu(raw, sc):
    a = collections.Counter()
    for r in raw:
        s = sc[r['file']]
        a.update(n=1, int_ok=s['int_ok'], tpv=s['tp_value'], tp=s['tp_v2'], nh=s['nh'], nr=s['nr'], full=s['full_v2'],
                 zero=s['status'] == 'ok' and s['nh'] == 0, parse_fail=s['status'] != 'ok',
                 refusal=s['status'] == 'no_json' and bool(REFUSAL.search(r['raw_output'])),
                 oov_int=s['status'] == 'ok' and s['hyp']['intent'] not in INV['intents'], oov_type=len(s['oov']),
                 we=s['wer_err'], wr=s['wer_ref'])
    n = a['n']
    return dict(n=n, intent_strict_acc=a['int_ok'] / n, entity_f1_strict=f1(a['tp'], a['nh'], a['nr']),
                value_only_precision=a['tpv'] / max(a['nh'], 1), value_only_recall=a['tpv'] / max(a['nr'], 1),
                value_only_f1=f1(a['tpv'], a['nh'], a['nr']), full_semantic_acc=a['full'] / n, wer_vs_target=a['we'] / max(a['wr'], 1),
                zero_entity_outputs=a['zero'], parse_failures=a['parse_fail'], command_refusals=a['refusal'],
                out_of_inventory_intents=a['oov_int'], out_of_inventory_entity_types=a['oov_type'])


def share(rows):
    ot = sum(r['cat'] == 'only_target' for r in rows); oi = sum(r['cat'] == 'only_interferer' for r in rows)
    return ot, oi, (ot / (ot + oi) if ot + oi else None)


def cat_summary(rows):
    n = len(rows)
    ot, oi, sh = share(rows)
    return dict(n=n, **{c: sum(r['cat'] == c for r in rows) / n for c in CATS}, target_any=sum(r['t_any'] for r in rows) / n,
                target_all=sum(r['t_all'] for r in rows) / n, single_speaker=ot + oi, target_share=sh)


def main(run, manifest, p2dir, out):
    run, out = Path(run), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    rep = dict(rules='docs/c3_gate_v1.md', run=str(run))
    # ---------------- C1
    c1 = [json.loads(l) for l in open(run / 'C1' / 'raw.jsonl')]
    sc1 = {r['file']: score_item(r) for r in c1}
    H1 = {f: vals(s['hyp']['entities']) for f, s in sc1.items()}
    per_rec = collections.defaultdict(list)
    for g in L['groups']:
        VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
        for k, D_T, D_I in (('A', VA - VB, VB - VA), ('B', VB - VA, VA - VB)):
            j = judge(D_T, D_I, H1[g[k]['file']]); per_rec[g[k]['file']].append(dict(group=g['group'], **j))
    c1_rows = [dict(file=f, n_groups=len(v), target_any=sum(x['t_any'] for x in v) / len(v), target_all=sum(x['t_all'] for x in v) / len(v),
                    categories=[x['cat'] for x in v]) for f, v in sorted(per_rec.items())]
    p2 = {json.loads(l)['file']: json.loads(l) for l in open(Path(p2dir) / 'raw.jsonl')}
    common = sorted(set(p2) & set(sc1))
    p2sc = {f: score_item(p2[f]) for f in common}
    same_raw = [f for f in common if p2[f]['raw_output'] == next(r for r in c1 if r['file'] == f)['raw_output']]
    same_ent = [f for f in common if sorted((t, normalize_v2(v)) for t, v in p2sc[f]['hyp']['entities'])
                == sorted((t, normalize_v2(v)) for t, v in sc1[f]['hyp']['entities'])]
    rep['C1'] = dict(slu=slu(c1, sc1), recordings=len(c1_rows),
                     mean_target_any=float(np.mean([r['target_any'] for r in c1_rows])),
                     mean_target_all=float(np.mean([r['target_all'] for r in c1_rows])),
                     consistency_with_22011390_P2=dict(common_recordings=len(common), identical_raw_output=len(same_raw),
                                                        identical_parsed_entities=len(same_ent),
                                                        differing=[f for f in common if f not in same_raw]))
    with open(out / 'c1_per_recording.jsonl', 'w') as fh:
        for r in c1_rows:
            fh.write(json.dumps(r) + '\n')
    # ---------------- C3
    c3 = [json.loads(l) for l in open(run / 'C3' / 'raw.jsonl')]
    sc3 = {r['file']: score_item(r) for r in c3}
    gl = {g['group']: g for g in L['groups']}
    rows = []
    for r in c3:
        g = gl[r['group']]; t = r['target']; i = 'B' if t == 'A' else 'A'
        VT, VI = vals(g[t]['entities']), vals(g[i]['entities'])
        H = vals(sc3[r['file']]['hyp']['entities'])
        j = judge(VT - VI, VI - VT, H)
        jA = judge(vals(g['A']['entities']) - vals(g['B']['entities']), vals(g['B']['entities']) - vals(g['A']['entities']), H)
        who = dict(only_target='only_A', only_interferer='only_B', both='both', neither='neither')[jA['cat']]
        m = man[r['file']]
        rows.append(dict(key=r['file'], group=r['group'], swap=r['swap'], target=t, cat=j['cat'], t_any=j['t_any'], t_all=j['t_all'],
                         who=who, Hset=sorted(H.elements()), status=sc3[r['file']]['status'], stratum=g['rir']['stratum'],
                         t60=g['rir']['t60'], overlap=m['overlap_active_frames'], sir_full=m['sir_mic0_full_db']))
    by_g = collections.defaultdict(list)
    for x in rows:
        by_g[x['group']].append(x)
    gids = sorted(by_g)
    ot, oi, sh = share(rows)
    rng = np.random.default_rng(SEED); boots = []; skipped = 0
    for _ in range(N_BOOT):
        samp = [x for gi in rng.choice(gids, size=len(gids), replace=True) for x in by_g[gi]]
        s = share(samp)[2]
        if s is None:
            skipped += 1
        else:
            boots.append(s)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    if ot + oi < 50:
        gate = 'STOP: fewer than 50 single-speaker decisions (insufficient)'
    elif lo > 0.60:
        gate = 'STOP: CI lower bound > 0.60 (suspected leakage)'
    elif hi < 0.40:
        gate = 'STOP: CI upper bound < 0.40 (bias to interferer or data problem)'
    else:
        gate = 'PASS: may proceed to front-end inference'
    flip = sum(all(x['cat'] == 'only_target' for x in by_g[gi]) for gi in gids)
    changed_who = sum(by_g[gi][0]['who'] != by_g[gi][1]['who'] for gi in gids)
    same_vals = sum(by_g[gi][0]['Hset'] == by_g[gi][1]['Hset'] for gi in gids)
    who_pairs = collections.Counter(f"{by_g[gi][0]['who']}->{by_g[gi][1]['who']}" for gi in gids)
    strata = {}
    for k, bins in BINS.items():
        if bins is None:
            keys = sorted({x['stratum'] for x in rows}); sel = {b: [x for x in rows if x['stratum'] == b] for b in keys}
        else:
            sel = {f'[{lo_:g},{hi_:g})': [x for x in rows if lo_ <= x[k] < hi_] for lo_, hi_ in bins}
        strata[k] = {b: (cat_summary(v) if v else dict(n=0)) for b, v in sel.items()}
    rep['C3'] = dict(slu_vs_target=slu(c3, sc3), samples=len(rows), groups=len(gids), **cat_summary(rows),
                     target_share_ci95=[lo, hi], bootstrap=dict(n=N_BOOT, seed=SEED, unit='group (both swaps)', skipped=skipped),
                     gate=gate, pair_flip_accuracy=flip / len(gids), pair_flip_groups=flip,
                     answer_changed_by_who=changed_who, answer_who_transitions=dict(who_pairs.most_common()),
                     identical_entity_values_across_swaps=same_vals, strata_exploratory=strata)
    with open(out / 'c3_per_sample.jsonl', 'w') as fh:
        for x in rows:
            fh.write(json.dumps(x) + '\n')
    (out / 'eval_c3_summary.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
