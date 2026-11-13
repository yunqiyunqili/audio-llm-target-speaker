"""Offline re-scoring of the C1 semantic pilot from raw.jsonl only (no GPU, no re-generation).

Keeps the strict metrics and adds:
  * entity value match ignoring type (multiset of normalised values)
  * entity token-overlap match: same type and >= 1 shared normalised token (one-to-one, greedy in ref order)
  * intent accuracy on the scenario prefix (text before the first '_')
  * parse status counts, dropped entities, entity types outside the inventory, outputs that hit max_new_tokens
  * every item with any error: ref sentence, ref labels, raw output, automatic error types
Automatic error types: parse, transcript (normalised WER > 0), intent, entity_boundary (same type, token
overlap, not exact), entity_type (same normalised value, different type), entity_missing / entity_spurious
(unmatched otherwise). 'annotation issue' is NOT assigned automatically; it is a manual judgement.
Usage: python -I scripts/qwen/rescore_c1_semantic.py RUN_DIR [MAX_NEW_TOKENS]
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slu_normalize import entity_counts, normalize, parse, wer_counts  # noqa: E402

INV = json.load(open(PROJ / 'configs/slurp_label_inventory_train.json'))


def f1(tp, h, r):
    return 2 * tp / (h + r) if h + r else 0.0


def overlap_match(ref, hyp, need_type=True):
    R = [(t, set(normalize(v).split())) for t, v in ref]; H = [(t, set(normalize(v).split())) for t, v in hyp]
    used = set(); tp = 0
    for t, toks in R:
        for j, (th, hk) in enumerate(H):
            if j not in used and (th == t or not need_type) and toks & hk:
                used.add(j); tp += 1; break
    return tp


def main(run_dir, max_new=256):
    run = Path(run_dir)
    items = [json.loads(l) for l in open(run / 'raw.jsonl')]
    agg = collections.Counter(); status = collections.Counter(); errors = []
    for it in items:
        obj, st = parse(it['raw_output'])
        hyp = obj or dict(transcript='', intent='', entities=[], dropped_entities=0)
        ref = [tuple(x) for x in it['ref_entities']]; H = hyp['entities']
        status[st] += 1
        tp, nh, nr = entity_counts(ref, H)
        tpv, _, _ = entity_counts([('x', v) for _, v in ref], [('x', v) for _, v in H])
        tpo = overlap_match(ref, H)
        e, n = wer_counts(normalize(it['ref_sentence']), normalize(hyp['transcript']))
        int_ok = hyp['intent'] == it['ref_intent']
        scen_ok = hyp['intent'].split('_')[0] == it['ref_intent'].split('_')[0] if hyp['intent'] else False
        oov_types = [t for t, _ in H if t not in INV['entity_types']]
        hit_max = it['new_tokens'] >= max_new
        agg.update(n=1, ent_tp=tp, ent_hyp=nh, ent_ref=nr, ent_tp_value=tpv, ent_tp_overlap=tpo, wer_err=e, wer_ref=n,
                   intent_ok=int_ok, scenario_ok=scen_ok, intent_oov=hyp['intent'] not in INV['intents'],
                   dropped=hyp.get('dropped_entities', 0), oov_types=len(oov_types), hit_max=hit_max)
        # automatic error types
        types = []
        if st != 'ok':
            types.append('parse')
        if e > 0:
            types.append('transcript')
        if not int_ok:
            types.append('intent')
        Rn = [(t, normalize(v)) for t, v in ref]; Hn = [(t, normalize(v)) for t, v in H]
        exact = [x for x in Rn if x in Hn]
        for (t, v) in Rn:
            if (t, v) in exact:
                continue
            same_val_other_type = any(hv == v and ht != t for ht, hv in Hn)
            boundary = any(ht == t and hv != v and set(hv.split()) & set(v.split()) for ht, hv in Hn)
            types.append('entity_type' if same_val_other_type else 'entity_boundary' if boundary else 'entity_missing')
        for (t, v) in Hn:
            if (t, v) not in Rn and not any(rv == v or (rt == t and set(rv.split()) & set(v.split())) for rt, rv in Rn):
                types.append('entity_spurious')
        if types:
            errors.append(dict(file=it['file'], slurp_id=it['slurp_id'], ref_sentence=it['ref_sentence'],
                               ref_intent=it['ref_intent'], ref_entities=it['ref_entities'], raw_output=it['raw_output'],
                               hyp_intent=hyp['intent'], hyp_entities=H, normalised_ref=normalize(it['ref_sentence']),
                               normalised_hyp=normalize(hyp['transcript']), error_types=sorted(set(types)),
                               error_type_counts=dict(collections.Counter(types)), oov_entity_types=oov_types,
                               new_tokens=it['new_tokens']))
    n = agg['n']
    summary = dict(
        n=n, parse_status=dict(status), dropped_entities=agg['dropped'], oov_entity_types=agg['oov_types'],
        intent_out_of_inventory=agg['intent_oov'], hit_max_new_tokens=agg['hit_max'], max_new_tokens=max_new,
        strict=dict(intent_accuracy=agg['intent_ok'] / n, wer=agg['wer_err'] / max(agg['wer_ref'], 1),
                    entity_precision=agg['ent_tp'] / max(agg['ent_hyp'], 1), entity_recall=agg['ent_tp'] / max(agg['ent_ref'], 1),
                    entity_f1=f1(agg['ent_tp'], agg['ent_hyp'], agg['ent_ref']),
                    counts=dict(tp=agg['ent_tp'], hyp=agg['ent_hyp'], ref=agg['ent_ref'], intent_correct=agg['intent_ok'])),
        relaxed=dict(scenario_accuracy=agg['scenario_ok'] / n, scenario_correct=agg['scenario_ok'],
                     entity_f1_value_only=f1(agg['ent_tp_value'], agg['ent_hyp'], agg['ent_ref']), tp_value_only=agg['ent_tp_value'],
                     entity_f1_token_overlap=f1(agg['ent_tp_overlap'], agg['ent_hyp'], agg['ent_ref']), tp_token_overlap=agg['ent_tp_overlap']),
        items_with_errors=len(errors),
        error_type_items=dict(collections.Counter(t for e in errors for t in e['error_types'])))
    (run / 'rescore_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with open(run / 'rescore_errors.jsonl', 'w') as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256)
