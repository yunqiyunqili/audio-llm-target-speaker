"""Descriptive: date entities in the devel selection / report halves (docs/lora_slu_plan_v2.md §3.2) and C0-428 date misses.
Reads devel_recordings.csv and the C0-428 eval entity classes (recomputed with eval_c0_devel428.classify, unchanged). No split change.
Usage: python -I -B scripts/qwen/c0_date_halves_diag.py C0_428_RAW OUT_DIR
"""
import collections
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c0_devel428 import classify  # noqa: E402
from slu_normalize import parse  # noqa: E402


def main(raw, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    R = {json.loads(l)['file']: json.loads(l) for l in open(raw)}
    ids = sorted({r['slurp_id'] for r in R.values()}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    rep = {}
    for half, keep in (('selection', lambda r: r['slurp_id'] in sel), ('report', lambda r: r['slurp_id'] not in sel), ('all', lambda r: True)):
        rows = []
        for f, r in sorted(R.items()):
            if not keep(r):
                continue
            cl = classify(r['ref_entities'], r['raw_output'], r['ref_sentence'])
            obj, _ = parse(r['raw_output'])
            for (t, v), (_, c) in zip(r['ref_entities'], cl):
                if t == 'date':
                    rows.append(dict(file=f, slurp_id=r['slurp_id'], value=v, cls=c, sentence=r['ref_sentence'], intent=r['ref_intent'],
                                     pred=[list(p) for p in obj['entities']] if obj else None))
        n = len(rows); hit = sum(x['cls'] == 'hit' for x in rows)
        vc = collections.Counter(x['value'] for x in rows)
        rate = {v: dict(n=k, hit=sum(x['cls'] == 'hit' for x in rows if x['value'] == v)) for v, k in vc.most_common(12)}
        rep[half] = dict(date_entities=n, sentences_with_date=len({x['slurp_id'] for x in rows}), recordings_with_date=len({x['file'] for x in rows}),
                         recall=hit / n if n else None, classes=dict(collections.Counter(x['cls'] for x in rows)),
                         top_values=rate, by_intent={k: dict(n=v, hit=sum(x['cls'] == 'hit' for x in rows if x['intent'] == k))
                                                    for k, v in collections.Counter(x['intent'] for x in rows).most_common(8)},
                         distinct_values=len(vc), single_word_share=sum(len(x['value'].split()) == 1 for x in rows) / n if n else None,
                         recall_single_word=(sum(x['cls'] == 'hit' for x in rows if len(x['value'].split()) == 1) /
                                             max(1, sum(len(x['value'].split()) == 1 for x in rows))),
                         recall_multi_word=(sum(x['cls'] == 'hit' for x in rows if len(x['value'].split()) > 1) /
                                            max(1, sum(len(x['value'].split()) > 1 for x in rows))),
                         missed_examples=[x for x in rows if x['cls'] != 'hit'][:15],
                         recall_by_recording_dedup=None)
        # recall counting each sentence once (duplicates removed)
        seen = {}
        for x in rows:
            seen.setdefault((x['slurp_id'], x['value']), x['cls'] == 'hit')
        rep[half]['recall_by_unique_sentence'] = sum(seen.values()) / len(seen) if seen else None
        rep[half]['unique_sentence_date_entities'] = len(seen)
    (out / 'c0_date_halves_diag.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for h in ('selection', 'report', 'all'):
        x = rep[h]
        print('==', h, {k: x[k] for k in ('date_entities', 'sentences_with_date', 'recordings_with_date', 'recall', 'recall_by_unique_sentence', 'unique_sentence_date_entities', 'classes', 'distinct_values', 'single_word_share', 'recall_single_word', 'recall_multi_word')})
        print('  top', json.dumps(x['top_values'], ensure_ascii=False))
        print('  intent', json.dumps(x['by_intent']))
        if h != 'all':
            for e in x['missed_examples']:
                print('   MISS', e['cls'][:6], repr(e['value']), '|', e['sentence'], '|', e['pred'])


if __name__ == '__main__':
    main(*sys.argv[1:3])
