"""Evaluate C4 v1 per docs/c4_v1.md (frozen): C4 (two front-ends), C3 (22014089) and R1 (22014155) on the same 200 samples.
slu-score-v2 / binding-score-v1 imported unchanged. Usage:
  python -I scripts/qwen/eval_c4_v1.py C4_RUN C3_RAW R1_RAW MIX_MANIFEST OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import parse, wer_counts  # noqa: E402
from binding_score_v1 import judge, vals  # noqa: E402
from postproc_c1_ext import REFUSAL  # noqa: E402

CATS = ('only_target', 'only_interferer', 'both', 'neither')
BINS = dict(stratum=None, t60=[(0.2, 0.3), (0.3, 0.4), (0.4, 0.5000001)], overlap=[(0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0000001)],
            sir_full=[(-1e9, -0.5), (-0.5, 0.5000001), (0.5000001, 1e9)])


def wer(ref, hyp):
    e, n = wer_counts(normalize_v2(ref), normalize_v2(hyp)); return e / n


def score(raw_by_key, L, man, single=False):
    rows = []
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'
        r = raw_by_key[key]
        obj, st = parse(r['raw_output'])
        tr = obj['transcript'] if obj else ''
        H = vals(obj['entities']) if obj else collections.Counter()
        VT, VI = vals(g[t]['entities']), vals(g[i]['entities'])
        j = judge(VT - VI, VI - VT, H)
        wt = wer(g[t]['sentence'], tr); wi = None if single else wer(g[i]['sentence'], tr)
        refv = {normalize_v2(v) for _, v in g[t]['entities'] + ([] if single else g[i]['entities'])}
        wmin = wt if single else min(wt, wi)
        rows.append(dict(key=key, group=m['group'], swap=m['swap'], cat=j['cat'], t_any=j['t_any'], t_all=j['t_all'], M=wmin < 0.5,
                         U=not (set(H) & refv) and wmin >= 0.5, parse_fail=st != 'ok',
                         refusal=st == 'no_json' and bool(REFUSAL.search(r['raw_output'])),
                         stratum=g['rir']['stratum'], t60=g['rir']['t60'], overlap=m['overlap_active_frames'], sir_full=m['sir_mic0_full_db']))
    return rows


def summ(rows):
    n = len(rows)
    if not n:
        return dict(n=0)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['group']].append(r)
    full_groups = [v for v in by.values() if len(v) == 2]
    return dict(n=n, **{c: sum(r['cat'] == c for r in rows) / n for c in CATS}, target_any=sum(r['t_any'] for r in rows) / n,
                target_all=sum(r['t_all'] for r in rows) / n,
                pair_flip_accuracy=(sum(all(x['cat'] == 'only_target' for x in v) for v in full_groups) / len(full_groups)) if full_groups else None,
                M=sum(r['M'] for r in rows), U=sum(r['U'] for r in rows), parse_fail=sum(r['parse_fail'] for r in rows),
                refusal=sum(r['refusal'] for r in rows))


def main(c4, c3raw, r1raw, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    R = {}
    for fe in ('blstm_mvdr', 'fasnet_tac'):
        R[f'C4_{fe}'] = score({json.loads(l)['file']: json.loads(l) for l in open(Path(c4) / fe / 'raw.jsonl')}, L, man)
    R['C3'] = score({json.loads(l)['file']: json.loads(l) for l in open(c3raw)}, L, man)
    r1 = {json.loads(l)['file']: json.loads(l) for l in open(r1raw)}
    r1_by_key = {k: r1[L[m['group']][m['target']]['file']] for k, m in man.items()}
    R['R1'] = score(r1_by_key, L, man, single=True)
    rep = dict(spec='docs/c4_v1.md', conditions={}, decisions={})
    for c, rows in R.items():
        st = {}
        for k, bins in BINS.items():
            if bins is None:
                st[k] = {s: summ([r for r in rows if r['stratum'] == s]) for s in sorted({r['stratum'] for r in rows})}
            else:
                st[k] = {f'[{lo:g},{hi:g})': summ([r for r in rows if lo <= r[k] < hi]) for lo, hi in bins}
        rep['conditions'][c] = dict(all=summ(rows), strata=st)
    a = {c: v['all']['target_any'] for c, v in rep['conditions'].items()}
    for fe in ('blstm_mvdr', 'fasnet_tac'):
        c = f'C4_{fe}'; d_c3 = a[c] - a['C3']; gap = a['R1'] - a[c]
        rep['decisions'][fe] = dict(delta_vs_C3=d_c3, separation_helps=d_c3 >= 0.15 - 1e-12, gap_to_R1=gap,
                                    distortion='not the main obstacle' if gap <= 0.10 + 1e-12 else 'still the main obstacle')
    (out / 'eval_c4_v1.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(out / 'per_sample.jsonl', 'w') as fh:
        for c, rows in R.items():
            for r in rows:
                fh.write(json.dumps(dict(cond=c, **r)) + '\n')
    for c, v in rep['conditions'].items():
        print(c, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v['all'].items()}))
    print(json.dumps(rep['decisions'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
