"""SLU scoring rules v2 (frozen 2026-11-05, docs/scoring_rules.md). Offline, from raw.jsonl only; never re-generates.

Does NOT overwrite v1 outputs (rescore_summary.json / rescore_errors.jsonl); writes rescore_v2_summary.json and
rescore_v2_errors.jsonl next to raw.jsonl. v1 strict metrics are reported unchanged under "strict_v1".
Changes vs v1 (scripts/qwen/rescore_c1_semantic.py):
  * normalize_v2 = slu_normalize.normalize followed by: "rnb" / "r'n'b" / "r and b" / "r n b" -> "r n b"
  * possessive tokens: tokens compared with a trailing "'s" removed when deciding boundary / token overlap
    ("derek's" vs "derek" -> entity_boundary, counted in token-overlap F1; NOT counted in strict F1)
  * WER uses normalize_v2 ("wer"); WER with v1 normalisation kept as "wer_v1"
  * full semantic accuracy: parse ok AND intent exact AND entity multiset {(type, normalize_v2(value))} identical;
    transcript ignored. Also reported with v1 normalisation.
  * items listed in a disputed-items file are flagged review_status = "待确认" (scores are NOT changed)
No manual entity-type mapping. Usage:
  python -I scripts/qwen/rescore_slu_v2.py RUN_DIR [--no-wer] [--max-new-tokens 256] [--disputed configs/c1_disputed_items.json]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slu_normalize import normalize, parse, wer_counts  # noqa: E402

INV = json.load(open(PROJ / 'configs/slurp_label_inventory_train.json'))
VERSION = 'slu-score-v2'


def normalize_v2(s):
    return re.sub(r"\b(?:rnb|r'n'b|r and b|r n b)\b", 'r n b', normalize(s))


def toks(v):
    return {re.sub(r"'s$", '', t) for t in v.split()}


def multiset_tp(R, H):
    pool = list(R); tp = 0
    for x in H:
        if x in pool:
            pool.remove(x); tp += 1
    return tp


def overlap_tp(R, H, need_type=True):
    used = set(); tp = 0
    for t, v in R:
        for j, (th, hv) in enumerate(H):
            if j not in used and (th == t or not need_type) and toks(v) & toks(hv):
                used.add(j); tp += 1; break
    return tp


def f1(tp, h, r):
    return 2 * tp / (h + r) if h + r else 0.0


def classify(Rn, Hn):
    types = []
    pool = list(Hn)
    for x in Rn:
        if x in pool:
            pool.remove(x)
    Rleft = list(Rn)
    for x in Hn:
        if x in Rleft:
            Rleft.remove(x)
    for (t, v) in Rleft:
        if any(hv == v and ht != t for ht, hv in pool):
            types.append('entity_type')
        elif any(ht == t and toks(hv) & toks(v) for ht, hv in pool):
            types.append('entity_boundary')
        else:
            types.append('entity_missing')
    for (t, v) in pool:
        if not any(rv == v or (rt == t and toks(rv) & toks(v)) for rt, rv in Rleft):
            types.append('entity_spurious')
    return types


def score_item(it):
    obj, st = parse(it['raw_output'])
    hyp = obj or dict(transcript='', intent='', entities=[], dropped_entities=0)
    ref = [tuple(x) for x in it['ref_entities']]; H = [tuple(x) for x in hyp['entities']]
    R1 = [(t, normalize(v)) for t, v in ref]; H1 = [(t, normalize(v)) for t, v in H]
    R2 = [(t, normalize_v2(v)) for t, v in ref]; H2 = [(t, normalize_v2(v)) for t, v in H]
    int_ok = st == 'ok' and hyp['intent'] == it['ref_intent']
    scen_ok = st == 'ok' and bool(hyp['intent']) and hyp['intent'].split('_')[0] == it['ref_intent'].split('_')[0]
    e, n = wer_counts(normalize_v2(it['ref_sentence']), normalize_v2(hyp['transcript']))
    e1, _ = wer_counts(normalize(it['ref_sentence']), normalize(hyp['transcript']))
    return dict(status=st, hyp=hyp, int_ok=int_ok, scen_ok=scen_ok, nh=len(H), nr=len(ref),
                tp_v1=multiset_tp(R1, H1), tp_v2=multiset_tp(R2, H2),
                tp_value=multiset_tp([v for _, v in R2], [v for _, v in H2]), tp_overlap=overlap_tp(R2, H2),
                full_v1=int_ok and sorted(R1) == sorted(H1), full_v2=int_ok and sorted(R2) == sorted(H2),
                wer_err=e, wer_err_v1=e1, wer_ref=n, R2=R2, H2=H2, oov=[t for t, _ in H if t not in INV['entity_types']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--no-wer', action='store_true', help='text-input condition: transcript WER not reported')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--disputed', type=Path, default=PROJ / 'configs/c1_disputed_items.json')
    a = ap.parse_args()
    disputed = {d['file']: d['reason'] for d in json.load(open(a.disputed))['items']} if a.disputed.exists() else {}
    items = [json.loads(l) for l in open(a.run_dir / 'raw.jsonl')]
    agg = collections.Counter(); status = collections.Counter(); errors = []; per_item = []
    for it in items:
        s = score_item(it)
        status[s['status']] += 1
        agg.update(n=1, int_ok=s['int_ok'], scen_ok=s['scen_ok'], nh=s['nh'], nr=s['nr'], tp_v1=s['tp_v1'], tp_v2=s['tp_v2'],
                   tp_value=s['tp_value'], tp_overlap=s['tp_overlap'], full_v1=s['full_v1'], full_v2=s['full_v2'],
                   wer_err=s['wer_err'], wer_err_v1=s['wer_err_v1'], wer_ref=s['wer_ref'], oov=len(s['oov']), dropped=s['hyp'].get('dropped_entities', 0),
                   intent_oov=s['status'] == 'ok' and s['hyp']['intent'] not in INV['intents'],
                   hit_max=it['new_tokens'] >= a.max_new_tokens)
        agg['seconds_x1e6'] += int(round(it.get('seconds', 0) * 1e6))
        types = (['parse'] if s['status'] != 'ok' else []) + (['transcript'] if s['wer_err'] and not a.no_wer else []) + \
                (['intent'] if not s['int_ok'] else []) + classify(s['R2'], s['H2'])
        rec = dict(file=it['file'], slurp_id=it['slurp_id'], int_ok=s['int_ok'], scen_ok=s['scen_ok'], full_v2=s['full_v2'],
                   tp_v2=s['tp_v2'], nh=s['nh'], nr=s['nr'], hyp_intent=s['hyp']['intent'], hyp_entities=s['hyp']['entities'],
                   error_types=sorted(set(types)))
        per_item.append(rec)
        if types:
            errors.append(dict(file=it['file'], slurp_id=it['slurp_id'], ref_sentence=it['ref_sentence'], ref_intent=it['ref_intent'],
                               ref_entities=it['ref_entities'], raw_output=it['raw_output'], hyp_intent=s['hyp']['intent'],
                               hyp_entities=s['hyp']['entities'], parse_status=s['status'],
                               normalised_ref=normalize_v2(it['ref_sentence']), normalised_hyp=normalize_v2(s['hyp']['transcript']),
                               error_types=sorted(set(types)), error_type_counts=dict(collections.Counter(types)),
                               oov_entity_types=s['oov'], new_tokens=it['new_tokens'],
                               review_status='待确认' if it['file'] in disputed else '', dispute_reason=disputed.get(it['file'], '')))
    n = agg['n']
    summary = dict(
        version=VERSION, n=n, parse_status=dict(status), dropped_entities=agg['dropped'], oov_entity_types=agg['oov'],
        intent_out_of_inventory=agg['intent_oov'], hit_max_new_tokens=agg['hit_max'], max_new_tokens=a.max_new_tokens,
        seconds_total=agg['seconds_x1e6'] / 1e6,
        intent_accuracy=agg['int_ok'] / n, scenario_accuracy=agg['scen_ok'] / n,
        strict_v1=dict(entity_f1=f1(agg['tp_v1'], agg['nh'], agg['nr']), tp=agg['tp_v1'], full_semantic_accuracy=agg['full_v1'] / n,
                       full_semantic_correct=agg['full_v1']),
        strict_v2=dict(entity_precision=agg['tp_v2'] / max(agg['nh'], 1), entity_recall=agg['tp_v2'] / max(agg['nr'], 1),
                       entity_f1=f1(agg['tp_v2'], agg['nh'], agg['nr']), tp=agg['tp_v2'],
                       full_semantic_accuracy=agg['full_v2'] / n, full_semantic_correct=agg['full_v2']),
        entity_f1_value_only=f1(agg['tp_value'], agg['nh'], agg['nr']), tp_value_only=agg['tp_value'],
        entity_f1_token_overlap=f1(agg['tp_overlap'], agg['nh'], agg['nr']), tp_token_overlap=agg['tp_overlap'],
        hyp_entities=agg['nh'], ref_entities=agg['nr'], intent_correct=agg['int_ok'], scenario_correct=agg['scen_ok'],
        wer=None if a.no_wer else agg['wer_err'] / max(agg['wer_ref'], 1), wer_counts=None if a.no_wer else [agg['wer_err'], agg['wer_ref']],
        wer_v1=None if a.no_wer else agg['wer_err_v1'] / max(agg['wer_ref'], 1),
        items_with_errors=len(errors), disputed_items_flagged=sum(e['review_status'] == '待确认' for e in errors),
        error_type_items=dict(collections.Counter(t for e in errors for t in e['error_types'])))
    (a.run_dir / 'rescore_v2_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    with open(a.run_dir / 'rescore_v2_errors.jsonl', 'w') as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + '\n')
    with open(a.run_dir / 'rescore_v2_items.jsonl', 'w') as fh:
        for r in per_item:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
