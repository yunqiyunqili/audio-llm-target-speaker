"""Separation-quality diagnostic, step 2 (docs/separation_quality_diag_v1.md, frozen; read-only; descriptive, no causal claim).
Joins metrics.jsonl with C5 FaSNet-TAC and C4-direct downstream scores (eval_c4_v1.score / summ unchanged).
Usage: python -I scripts/qwen/sep_quality_diag.py METRICS_JSONL C5_RUN C4_DIRECT_RUN MIX_MANIFEST OUT_JSON
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402

BINS = dict(sisdr_T=[(-1e9, 5), (5, 10), (10, 1e9)], irr_db=[(-1e9, 10), (10, 20), (20, 1e9)], rar_db=[(-1e9, 5), (5, 10), (10, 1e9)])
KEEP = ('n', 'only_target', 'only_interferer', 'neither', 'target_any', 'M', 'U')


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def two_by_two(keys, a, b):
    c = collections.Counter((a[k]['t_any'], b[k]['t_any']) for k in keys)
    return dict(both_hit=c[(True, True)], only_C4direct_hit=c[(False, True)], only_C5_hit=c[(True, False)], neither_hit=c[(False, False)])


def main(metrics, c5, c4d, manifest, out_json):
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    M = {json.loads(l)['key']: json.loads(l) for l in open(metrics)}
    r5 = {r['key']: r for r in score(load(Path(c5) / 'C5_fasnet_tac' / 'raw.jsonl'), L, man)}
    rd = {r['key']: r for r in score(load(Path(c4d) / 'C4_direct' / 'raw.jsonl'), L, man)}
    pick = lambda s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in KEEP}  # noqa: E731
    rep = dict(spec='docs/separation_quality_diag_v1.md', note='descriptive only; correlations are not causal',
               metric_summary={m: dict(min=min(x[m] for x in M.values()), median=sorted(x[m] for x in M.values())[100], max=max(x[m] for x in M.values()))
                               for m in BINS},
               overall=dict(C5=pick(summ(list(r5.values()))), C4_direct=pick(summ(list(rd.values()))), two_by_two=two_by_two(list(M), r5, rd)),
               strata={})
    for m, bins in BINS.items():
        rep['strata'][m] = {}
        for lo, hi in bins:
            keys = [k for k in M if lo <= M[k][m] < hi]
            rep['strata'][m][f'[{lo:g},{hi:g})'] = dict(C5=pick(summ([r5[k] for k in keys])), C4_direct=pick(summ([rd[k] for k in keys])),
                                                        two_by_two=two_by_two(keys, r5, rd))
    Path(out_json).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
