"""C1 error taxonomy v1 (docs/c1_error_taxonomy_v1.md, frozen). Reads existing raw outputs only; CPU.
Classifies every gold entity NOT hit (normalize_v2 value multiset, type-agnostic, as binding-score-v1) for P2 (job 22014089 C1)
and Whisper+T2 (job 22016221 T_C1 / W_C1) into a parse / b spelling / c boundary / d type / e missed-extraction / f misheard.
Diagnostic normalisation N() is local to this script. Also: target_any with class b counted as hit (diagnosis only).
Usage: python -I -B scripts/qwen/c1_error_taxonomy.py OUT_DIR
"""
import collections
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from binding_score_v1 import judge, vals  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import parse  # noqa: E402

O = PROJ / 'outputs'
INP = dict(P2=O / 'qwen_omni_c3_p2/22014089/C1/raw.jsonl', T2=O / 'qwen_omni_c7w_f1/22016221/T_C1/raw.jsonl', W=O / 'qwen_omni_c7w_f1/22016221/W_C1/raw.jsonl')
MANIFEST = '<SHARED_ROOT>/data/mix_v1/devel_seed2026/manifest.jsonl'
STORED_P2 = O / 'qwen_omni_c3_p2_eval/22014089/eval_c3_summary.json'
STORED_WT2 = O / 'c7w_f1_eval/22016732_full/eval_c7w_f1.json'
CLASSES = ('a_parse_fail', 'b_spelling', 'c_boundary', 'd_type', 'e_missed_extraction', 'f_misheard')
STOP = {'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for', 'my', 'me', 'and', 'is', 'it', 'this', 'that'}
UNITS = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = dict(twenty=20, thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90)
NUM = {w: i for i, w in enumerate(UNITS)}


def N(s):
    s = s.lower()
    s = re.sub(r'\ba\.\s?m\.?', 'am', s); s = re.sub(r'\bp\.\s?m\.?', 'pm', s)
    s = re.sub(r'\b(\d{1,2}):00\b', r'\1', s); s = re.sub(r'\b(\d{1,2}):(\d{2})\b', r'\1 \2', s)
    s = re.sub(r"\bo'?clock\b", ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    w = s.split(); out = []; i = 0
    while i < len(w):
        if w[i] in TENS:
            if i + 1 < len(w) and w[i + 1] in NUM and 1 <= NUM[w[i + 1]] <= 9:
                out.append(str(TENS[w[i]] + NUM[w[i + 1]])); i += 2; continue
            out.append(str(TENS[w[i]])); i += 1; continue
        out.append(str(NUM[w[i]]) if w[i] in NUM else w[i]); i += 1
    return ' '.join(out)


def same(a, b):
    A, B = N(a), N(b)
    if A == B:
        return True
    ap, bp = [x for x in A.split() if x not in ('am', 'pm')], [x for x in B.split() if x not in ('am', 'pm')]
    ha, hb = len(ap) != len(A.split()), len(bp) != len(B.split())
    clock = re.compile(r'\b\d{1,2}:\d{2}\b')
    return ap == bp and ((ha and not hb and bool(clock.search(b))) or (hb and not ha and bool(clock.search(a))))


def overlap(a, b):
    return set(N(a).split()) & set(N(b).split())


def contains(seq, sub):
    s, t = N(seq).split(), N(sub).split()
    return bool(t) and any(s[i:i + len(t)] == t for i in range(len(s) - len(t) + 1))


def load(p):
    if not Path(p).exists():
        sys.exit(f'STOP: missing {p}')
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(out):
    out = Path(out)
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']
    rec = {}
    for g in L:
        for k in 'AB':
            rec[g[k]['file']] = [tuple(e) for e in g[k]['entities']]
    R = {n: load(p) for n, p in INP.items()}
    for n, r in R.items():
        if set(r) != set(rec) or len(r) != 124:
            sys.exit(f'STOP: {n} has {len(r)} recordings / mismatch with mix_v1 124')
    for f in rec:
        assert R['T2'][f]['asr_text'] == R['W'][f]['raw_output'], f
        for n in ('P2', 'T2'):
            assert [tuple(e) for e in R[n][f]['ref_entities']] == rec[f], (n, f)
    out.mkdir(parents=True, exist_ok=False)
    # D_T membership (entity value distinguishing for target in >=1 group)
    dt = collections.defaultdict(collections.Counter)
    for g in L:
        VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
        dt[g['A']['file']] |= VA - VB; dt[g['B']['file']] |= VB - VA
    rep = dict(spec='docs/c1_error_taxonomy_v1.md', inputs={k: str(v) for k, v in INP.items()}, routes={})
    bhits = {}
    for route in ('P2', 'T2'):
        rows = []; bh = {}
        for f in sorted(rec):
            obj, st = parse(R[route][f]['raw_output'])
            pred = obj['entities'] if obj else []
            H = vals(pred) if obj else collections.Counter()
            trans = (obj['transcript'] if obj else '') if route == 'P2' else R['W'][f]['raw_output']
            left = H.copy(); dleft = dt[f].copy(); bh[f] = collections.Counter()
            for t, v in rec[f]:
                nv = normalize_v2(v)
                in_dt = dleft[nv] > 0
                if in_dt:
                    dleft[nv] -= 1
                if left[nv] > 0:
                    left[nv] -= 1; rows.append(dict(file=f, type=t, value=v, hit=True, in_dt=in_dt)); continue
                if st != 'ok':
                    c = 'a_parse_fail'
                elif any(pt.strip().lower() == t and same(v, pv) for pt, pv in pred):
                    c = 'b_spelling'; bh[f][nv] += 1
                elif any(pt.strip().lower() == t and overlap(v, pv) for pt, pv in pred):
                    c = 'c_boundary'
                elif any(pt.strip().lower() != t and (same(v, pv) or overlap(v, pv)) for pt, pv in pred):
                    c = 'd_type'
                elif contains(trans, v):
                    c = 'e_missed_extraction'
                else:
                    c = 'f_misheard'
                ov = [sorted(overlap(v, pv)) for pt, pv in pred if overlap(v, pv)]
                stop_only = c in ('c_boundary', 'd_type') and not any(same(v, pv) for pt, pv in pred) and all(set(o) <= STOP for o in ov)
                rows.append(dict(file=f, type=t, value=v, hit=False, in_dt=in_dt, cls=c, pred=[list(p) for p in pred], transcript=trans,
                                 parse_status=st, overlap_stopwords_only=stop_only))
        bhits[route] = bh

        def tab(rs):
            miss = [r for r in rs if not r['hit']]; cnt = collections.Counter(r['cls'] for r in miss)
            return dict(gold_entities=len(rs), hit=len(rs) - len(miss), missed=len(miss),
                        classes={c: dict(n=cnt[c], share_of_missed=cnt[c] / len(miss) if miss else None) for c in CLASSES},
                        cd_overlap_stopwords_only=sum(r.get('overlap_stopwords_only', False) for r in miss))
        ex = {c: [dict(file=r['file'], gold=[r['type'], r['value']], pred=r['pred'], transcript=r['transcript'])
                  for r in rows if not r['hit'] and r['cls'] == c][:10] for c in CLASSES}
        rep['routes'][route] = dict(all=tab(rows), dt_subset=tab([r for r in rows if r['in_dt']]),
                                    recordings=len(rec), recordings_with_entities=sum(bool(v) for v in rec.values()), examples=ex)
        with open(out / f'entities_{route}.jsonl', 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    # target_any with / without class b counted as hit
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(MANIFEST))}
    gl = {g['group']: g for g in L}
    for route in ('P2', 'T2'):
        Hb = {}
        for f in rec:
            obj, _ = parse(R[route][f]['raw_output']); Hb[f] = vals(obj['entities']) if obj else collections.Counter()
        res = {}
        for name, Hx in (('binding_v1', Hb), ('b_as_hit', {f: Hb[f] + bhits[route][f] for f in rec})):
            per = collections.defaultdict(list)
            for g in L:
                VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
                for k, D_T, D_I in (('A', VA - VB, VB - VA), ('B', VB - VA, VA - VB)):
                    per[g[k]['file']].append(judge(D_T, D_I, Hx[g[k]['file']])['t_any'])
            by_rec = sum(sum(v) / len(v) for v in per.values()) / len(per)
            s200 = []
            for key, m in man.items():
                g = gl[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'
                VT, VI = vals(g[t]['entities']), vals(g[i]['entities'])
                s200.append(judge(VT - VI, VI - VT, Hx[g[t]['file']])['t_any'])
            res[name] = dict(target_any_by_recording=by_rec, target_any_200=sum(s200) / len(s200))
        rep['routes'][route]['target_any'] = res
    p2s = json.load(open(STORED_P2))['C1']['mean_target_any']; wts = json.load(open(STORED_WT2))['conditions']['C7W_C1']['target_any']
    assert abs(rep['routes']['P2']['target_any']['binding_v1']['target_any_by_recording'] - p2s) < 1e-12, 'P2 consistency FAILED'
    assert abs(rep['routes']['T2']['target_any']['binding_v1']['target_any_200'] - wts) < 1e-12, 'W+T2 consistency FAILED'
    rep['consistency'] = dict(P2_by_recording=p2s, WT2_200=wts, passed=True)
    (out / 'c1_error_taxonomy.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for route in ('P2', 'T2'):
        x = rep['routes'][route]
        print(route, json.dumps({k: x['all'][k] for k in ('gold_entities', 'hit', 'missed')}),
              json.dumps({c: v['n'] for c, v in x['all']['classes'].items()}), 'DT', json.dumps({c: v['n'] for c, v in x['dt_subset']['classes'].items()}),
              x['dt_subset']['gold_entities'], x['dt_subset']['missed'], 'stoponly', x['all']['cd_overlap_stopwords_only'], json.dumps(x['target_any']))
    print('CONSISTENCY_PASS')


if __name__ == '__main__':
    main(sys.argv[1])
