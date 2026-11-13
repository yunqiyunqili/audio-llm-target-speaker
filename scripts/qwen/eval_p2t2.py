"""Evaluate P0, T0 (job 22010997) and P2, T2 (new run) with slu-score-v2 and binding-score-v1. Offline; writes a NEW dir.
Frozen code is imported unchanged: rescore_slu_v2.score_item / normalize_v2, binding_score_v1.qualified_pairs / vals / judge,
postproc_c1_ext.REFUSAL. Binding: same qualified pairs, same 76 tier-1 targets, per-recording then mean over recordings
(identical to binding_score_v1.main, which is hard-wired to P0/P1/T0 and is therefore not called).
Reported on 76 targets: target_any, target_all, only_target, neither. On 100 items: intent strict accuracy, value-only P/R/F1,
zero-entity outputs (parse ok, entities == []), command refusals (no_json + refusal regex), out-of-inventory intents and types.
Conditions whose raw.jsonl does not exist are skipped. Also applies the frozen P2 adoption rule (docs/p2_adoption_rule.md). Usage:
  python -I scripts/qwen/eval_p2t2.py DATA_DIR OLD_RUN NEW_RUN OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from rescore_slu_v2 import INV, f1, score_item  # noqa: E402
from binding_score_v1 import judge, qualified_pairs, vals  # noqa: E402
from postproc_c1_ext import REFUSAL  # noqa: E402


EPS = 1e-9   # float tolerance for >= / <= comparisons


def adoption(C):
    """Frozen P2 adoption rule (docs/p2_adoption_rule.md). Unrounded values; T2 is NOT used."""
    if not isinstance(C.get('P2'), dict) or not isinstance(C.get('P0'), dict):
        return 'P2 or P0 missing'
    p, q = C['P2'], C['P0']
    bp, bq = p['binding_entities'], q['binding_entities']
    inter = lambda b: b['only_interferer'] + b['both']  # noqa: E731
    rules = [
        ('R1 target_any >= 0.60', bp['target_any'], 0.60, bp['target_any'] >= 0.60 - EPS),
        ('R2 target_any >= P0 + 0.05', bp['target_any'], bq['target_any'] + 0.05, bp['target_any'] >= bq['target_any'] + 0.05 - EPS),
        ('R3 target_all >= P0 + 0.066', bp['target_all'], bq['target_all'] + 0.066, bp['target_all'] >= bq['target_all'] + 0.066 - EPS),
        ('R4 only_interferer+both <= P0 + 0.01', inter(bp), inter(bq) + 0.01, inter(bp) <= inter(bq) + 0.01 + EPS),
        ('R5 intent_strict_acc >= P0 - 0.03', p['intent_strict_acc'], q['intent_strict_acc'] - 0.03,
         p['intent_strict_acc'] >= q['intent_strict_acc'] - 0.03 - EPS),
        ('R6 value_only_precision >= P0 - 0.05', p['value_only_precision'], q['value_only_precision'] - 0.05,
         p['value_only_precision'] >= q['value_only_precision'] - 0.05 - EPS),
        ('R7 refusals <= 1 and parse_failures <= 2', [p['command_refusals'], p['parse_failures']], [1, 2],
         p['command_refusals'] <= 1 and p['parse_failures'] <= 2)]
    res = [dict(rule=r, value=v, threshold=t, passed=bool(ok)) for r, v, t, ok in rules]
    return dict(rules=res, adopt_P2=all(r['passed'] for r in res))


def main(data, old, new, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _, pairs = qualified_pairs(data)
    partners = collections.defaultdict(list)
    for a, b, g, dd, Da, Db in pairs:
        if g and dd <= 1.0 and Da and Db:
            partners[a].append((Da, Db)); partners[b].append((Db, Da))
    cfg = json.load(open(PROJ / 'configs/c1_ext_devel100.json'))
    targets = [it['file'] for it in cfg['items'] if it['tier'] == 1]
    src = dict(P0=Path(old) / 'P0', T0=Path(old) / 'T0', P2=Path(new) / 'P2', T2=Path(new) / 'T2')
    rep = dict(old_run=str(old), new_run=str(new), targets=len(targets), pairs=sum(len(partners[f]) for f in targets), conditions={})
    for c, d in src.items():
        if not (d / 'raw.jsonl').exists():
            rep['conditions'][c] = 'missing'; continue
        raw = [json.loads(l) for l in open(d / 'raw.jsonl')]
        assert [r['file'] for r in raw] == [it['file'] for it in cfg['items']], c
        sc = {r['file']: score_item(r) for r in raw}
        a = collections.Counter()
        for r in raw:
            s = sc[r['file']]
            a.update(int_ok=s['int_ok'], tpv=s['tp_value'], nh=s['nh'], nr=s['nr'],
                     zero=s['status'] == 'ok' and s['nh'] == 0, refusal=s['status'] == 'no_json' and bool(REFUSAL.search(r['raw_output'])),
                     no_json=s['status'] == 'no_json', parse_fail=s['status'] != 'ok', oov_int=s['status'] == 'ok' and s['hyp']['intent'] not in INV['intents'], oov_type=len(s['oov']))
        recs = []
        for f in targets:
            H = vals(sc[f]['hyp']['entities'])
            js = [judge(DT, DI, H) for DT, DI in partners[f]]
            n = len(js); cat = collections.Counter(j['cat'] for j in js)
            recs.append(dict(target_any=sum(j['t_any'] for j in js) / n, target_all=sum(j['t_all'] for j in js) / n,
                             only_target=cat['only_target'] / n, neither=cat['neither'] / n,
                             only_interferer=cat['only_interferer'] / n, both=cat['both'] / n))
        m = len(recs)
        rep['conditions'][c] = dict(
            n=len(raw), intent_strict_acc=a['int_ok'] / len(raw), value_only_precision=a['tpv'] / max(a['nh'], 1),
            value_only_recall=a['tpv'] / max(a['nr'], 1), value_only_f1=f1(a['tpv'], a['nh'], a['nr']),
            counts=dict(intent_correct=a['int_ok'], tp_value=a['tpv'], hyp_entities=a['nh'], ref_entities=a['nr']),
            zero_entity_outputs=a['zero'], command_refusals=a['refusal'], no_json=a['no_json'], parse_failures=a['parse_fail'],
            out_of_inventory_intents=a['oov_int'], out_of_inventory_entity_types=a['oov_type'],
            binding_entities={k: sum(r[k] for r in recs) / m for k in recs[0]}, binding_recordings=m)
    rep['p2_adoption'] = adoption(rep['conditions'])
    (out / 'eval_summary.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
