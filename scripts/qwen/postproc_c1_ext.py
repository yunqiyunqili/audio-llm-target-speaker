"""Post-processing of the C1 extension (job 22010997). Reads RUN/{P0,P1,T0}/raw.jsonl ONLY; writes to a NEW directory.
Uses the frozen scoring functions of scripts/qwen/rescore_slu_v2.py (imported, not modified); nothing in RUN is touched.
  1. command_refusal: parse_status == no_json AND the output refuses to carry out the command (regex below).
     Every no_json output is listed verbatim for manual review. In SLU scores it still counts as no_json.
  2. zero-entity outputs (parse ok and entities == []), and parse-failure outputs, per condition.
  3. P1 example values: normalize_v2 values of the 3 fixed examples' entities. Count parsed hyp entities whose
     normalize_v2(value) equals an example value while the item's reference values do not contain it. Same method
     for P0, P1 (and T0 for information). Reported: P1 - P0.
  4. Metrics per condition on: all 100; minus items whose reference values contain an example value; minus the
     items that overlap the 20-item pilot (configs/c1_semantic_devel20.json).
Usage: python -I scripts/qwen/postproc_c1_ext.py RUN_DIR OUT_DIR
"""
import collections
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_slu_v2 import f1, normalize_v2, score_item  # noqa: E402

CONDS = ('P0', 'P1', 'T0')
REFUSAL = re.compile(r"\b(sorry|i can't|i cannot|i can not|i'm unable|i am unable|i don't have access|as an ai)\b", re.I)


def metrics(scored, audio):
    n = len(scored); a = collections.Counter()
    for s in scored:
        a.update(int_ok=s['int_ok'], scen_ok=s['scen_ok'], nh=s['nh'], nr=s['nr'], tp=s['tp_v2'], tpv=s['tp_value'],
                 tpo=s['tp_overlap'], full=s['full_v2'], we=s['wer_err'], wr=s['wer_ref'], ok=s['status'] == 'ok')
    return dict(n=n, parse_ok=a['ok'], intent_acc=a['int_ok'] / n, scenario_acc=a['scen_ok'] / n,
                entity_f1_strict=f1(a['tp'], a['nh'], a['nr']), entity_f1_value_only=f1(a['tpv'], a['nh'], a['nr']),
                entity_f1_token_overlap=f1(a['tpo'], a['nh'], a['nr']), full_semantic_acc=a['full'] / n,
                counts=dict(intent=a['int_ok'], scenario=a['scen_ok'], tp=a['tp'], hyp=a['nh'], ref=a['nr'], full=a['full']),
                wer=a['we'] / max(a['wr'], 1) if audio else None)


def main(run, out):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.load(open(PROJ / 'configs/c1_ext_devel100.json'))
    ex_vals = sorted({normalize_v2(v) for e in cfg['examples_train'] for _, v in e['entities']})
    pilot = {it['file'] for it in json.load(open(PROJ / 'configs/c1_semantic_devel20.json'))['items']}
    items = {c: [json.loads(l) for l in open(run / c / 'raw.jsonl')] for c in CONDS}
    files = [r['file'] for r in items['P0']]
    assert all([r['file'] for r in items[c]] == files for c in CONDS)
    ref_vals = {r['file']: [normalize_v2(v) for _, v in r['ref_entities']] for r in items['P0']}
    ex_items = sorted(f for f in files if set(ref_vals[f]) & set(ex_vals))
    subsets = dict(all=set(files), minus_example_value_items=set(files) - set(ex_items), minus_pilot_overlap=set(files) - pilot)
    rep = dict(run=str(run), example_values=ex_vals, example_value_items=len(ex_items), pilot_overlap_items=len(set(files) & pilot),
               refusal_regex=REFUSAL.pattern, conditions={})
    no_json = []
    for c in CONDS:
        sc = {r['file']: score_item(r) for r in items[c]}
        refusals = 0
        for r in items[c]:
            if sc[r['file']]['status'] == 'no_json':
                ref = bool(REFUSAL.search(r['raw_output'])); refusals += ref
                no_json.append(dict(condition=c, file=r['file'], slurp_id=r['slurp_id'], ref_sentence=r['ref_sentence'],
                                    ref_intent=r['ref_intent'], command_refusal=ref, raw_output=r['raw_output']))
        leak = collections.Counter(); leak_items = []
        for r in items[c]:
            for t, v in sc[r['file']]['hyp']['entities']:
                nv = normalize_v2(v)
                if nv in ex_vals and nv not in ref_vals[r['file']]:
                    leak[nv] += 1; leak_items.append(dict(file=r['file'], slurp_id=r['slurp_id'], type=t, value=v))
        rep['conditions'][c] = dict(
            parse_status=dict(collections.Counter(s['status'] for s in sc.values())), command_refusal=refusals,
            zero_entity_outputs_parse_ok=sum(s['status'] == 'ok' and s['nh'] == 0 for s in sc.values()),
            zero_entity_outputs_incl_parse_fail=sum(s['nh'] == 0 for s in sc.values()),
            example_value_not_in_ref=sum(leak.values()), example_value_not_in_ref_by_value=dict(leak), example_value_hits=leak_items,
            metrics={k: metrics([sc[f] for f in files if f in v], c != 'T0') for k, v in subsets.items()})
    rep['P1_minus_P0_example_value_not_in_ref'] = (rep['conditions']['P1']['example_value_not_in_ref']
                                                   - rep['conditions']['P0']['example_value_not_in_ref'])
    rep['example_value_item_files'] = ex_items
    (out / 'postproc_summary.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    with open(out / 'no_json_outputs.jsonl', 'w') as fh:
        for x in no_json:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'conditions'}, ensure_ascii=False))
    for c in CONDS:
        d = rep['conditions'][c]
        print(c, json.dumps({k: d[k] for k in ('parse_status', 'command_refusal', 'zero_entity_outputs_parse_ok',
                                               'zero_entity_outputs_incl_parse_fail', 'example_value_not_in_ref',
                                               'example_value_not_in_ref_by_value')}))
        for k, m in d['metrics'].items():
            print('  ', k, json.dumps({x: (round(y, 3) if isinstance(y, float) else y) for x, y in m.items()}))
    for x in no_json:
        print('NO_JSON', x['condition'], x['slurp_id'], 'refusal' if x['command_refusal'] else 'not_refusal', '|', x['ref_sentence'], '|', repr(x['raw_output']))


if __name__ == '__main__':
    main(*sys.argv[1:3])
