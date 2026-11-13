"""Per-item differences between C1-extension conditions, from RUN_DIR/{P0,P1,T0}/rescore_v2_items.jsonl (offline).
Writes RUN_DIR/condition_diff.jsonl (one line per item: per-condition intent / full_v2 / tp / hyp intent / error types)
and RUN_DIR/condition_diff_summary.json (pattern counts over items, e.g. full_v2 "P0=0,P1=1,T0=1").
Usage: python -I scripts/qwen/compare_c1_ext_conditions.py RUN_DIR
"""
import collections
import json
import sys
from pathlib import Path

CONDS = ('P0', 'P1', 'T0')


def main(run):
    run = Path(run)
    rows = {c: {r['file']: r for r in map(json.loads, open(run / c / 'rescore_v2_items.jsonl'))} for c in CONDS}
    files = list(rows['P0'])
    assert all(set(rows[c]) == set(files) for c in CONDS)
    out, pat = [], {k: collections.Counter() for k in ('int_ok', 'scen_ok', 'full_v2')}
    for f in files:
        d = dict(file=f, slurp_id=rows['P0'][f]['slurp_id'], nr=rows['P0'][f]['nr'])
        for c in CONDS:
            r = rows[c][f]
            d[c] = dict(int_ok=r['int_ok'], full_v2=r['full_v2'], tp_v2=r['tp_v2'], nh=r['nh'], hyp_intent=r['hyp_intent'],
                        hyp_entities=r['hyp_entities'], error_types=r['error_types'])
        for k in pat:
            pat[k][','.join(f'{c}={int(rows[c][f][k])}' for c in CONDS)] += 1
        d['differs'] = len({json.dumps([rows[c][f]['int_ok'], rows[c][f]['full_v2'], rows[c][f]['tp_v2']]) for c in CONDS}) > 1
        out.append(d)
    with open(run / 'condition_diff.jsonl', 'w') as fh:
        for d in out:
            fh.write(json.dumps(d, ensure_ascii=False) + '\n')
    summ = dict(n=len(files), items_differing=sum(d['differs'] for d in out),
                patterns={k: dict(sorted(v.items())) for k, v in pat.items()})
    (run / 'condition_diff_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
