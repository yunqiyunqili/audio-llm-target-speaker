"""Evaluate C0 v1 (docs/c0_plan_v1.md, rules frozen): binding-score-v1 (per recording as eval_c3_v1 C1; 200-sample mapping as
eval_c7w_f1 C7W-C1), slu-score-v2, c1-error-taxonomy-v1 classification (functions imported unchanged; transcript = reference
sentence), overlap check vs job 22011390 T2, and the pre-registered reading. Reads raw outputs only.
Usage: python -I -B scripts/qwen/eval_c0_v1.py C0_RAW OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from binding_score_v1 import judge, vals  # noqa: E402
from eval_c3_v1 import slu  # noqa: E402
from c1_error_taxonomy import CLASSES, STOP, contains, overlap, same  # noqa: E402
from rescore_slu_v2 import normalize_v2, score_item  # noqa: E402
from slu_normalize import parse  # noqa: E402

O = PROJ / 'outputs'
MANIFEST = '<SHARED_ROOT>/data/mix_v1/devel_seed2026/manifest.jsonl'
OLD_T2 = O / 'qwen_omni_c1_p2t2/22011390/T2/raw.jsonl'
C1_P2 = O / 'qwen_omni_c3_p2_eval/22014089/eval_c3_summary.json'
TAX = O / 'c1_error_taxonomy_v1/c1_error_taxonomy.json'


def main(raw, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']
    rec = {g[k]['file']: [tuple(e) for e in g[k]['entities']] for g in L for k in 'AB'}
    sent = {g[k]['file']: g[k]['sentence'] for g in L for k in 'AB'}
    R = {json.loads(l)['file']: json.loads(l) for l in open(raw)}
    assert set(R) == set(rec) and len(R) == 124, len(R)
    for f in rec:
        assert R[f]['ref_sentence'] == sent[f] and [tuple(e) for e in R[f]['ref_entities']] == rec[f]
    rep = dict(spec='docs/c0_plan_v1.md', raw=str(raw))
    # slu-score-v2
    S = {f: score_item(R[f]) for f in sorted(R)}
    rep['slu_v2'] = slu(list(R.values()), S)
    # binding-score-v1
    H = {f: vals(S[f]['hyp']['entities']) for f in R}
    per = collections.defaultdict(list)
    for g in L:
        VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
        for k, D_T, D_I in (('A', VA - VB, VB - VA), ('B', VB - VA, VA - VB)):
            per[g[k]['file']].append(judge(D_T, D_I, H[g[k]['file']]))
    by_rec_any = sum(sum(x['t_any'] for x in v) / len(v) for v in per.values()) / len(per)
    by_rec_all = sum(sum(x['t_all'] for x in v) / len(v) for v in per.values()) / len(per)
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(MANIFEST))}; gl = {g['group']: g for g in L}
    s200 = []
    for m in man.values():
        g = gl[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'
        VT, VI = vals(g[t]['entities']), vals(g[i]['entities']); s200.append(judge(VT - VI, VI - VT, H[g[t]['file']]))
    c1p2 = json.load(open(C1_P2))['C1']['mean_target_any']
    rep['binding_v1'] = dict(target_any_by_recording=by_rec_any, target_all_by_recording=by_rec_all,
                             target_any_200=sum(x['t_any'] for x in s200) / 200, target_all_200=sum(x['t_all'] for x in s200) / 200,
                             only_interferer_200=sum(x['cat'] == 'only_interferer' for x in s200) / 200,
                             C1_P2_by_recording=c1p2, delta_C0_minus_C1P2=by_rec_any - c1p2)
    # taxonomy (same rules as c1-error-taxonomy-v1; transcript = reference sentence)
    rows = []
    for f in sorted(rec):
        obj, st = parse(R[f]['raw_output']); pred = obj['entities'] if obj else []
        left = vals(pred) if obj else collections.Counter(); trans = sent[f]
        for t, v in rec[f]:
            nv = normalize_v2(v)
            if left[nv] > 0:
                left[nv] -= 1; rows.append(dict(file=f, type=t, value=v, hit=True)); continue
            if st != 'ok':
                c = 'a_parse_fail'
            elif any(pt.strip().lower() == t and same(v, pv) for pt, pv in pred):
                c = 'b_spelling'
            elif any(pt.strip().lower() == t and overlap(v, pv) for pt, pv in pred):
                c = 'c_boundary'
            elif any(pt.strip().lower() != t and (same(v, pv) or overlap(v, pv)) for pt, pv in pred):
                c = 'd_type'
            elif contains(trans, v):
                c = 'e_missed_extraction'
            else:
                c = 'f_misheard'
            ov = [sorted(overlap(v, pv)) for pt, pv in pred if overlap(v, pv)]
            so = c in ('c_boundary', 'd_type') and not any(same(v, pv) for pt, pv in pred) and all(set(o) <= STOP for o in ov)
            rows.append(dict(file=f, type=t, value=v, hit=False, cls=c, pred=[list(p) for p in pred], transcript=trans, overlap_stopwords_only=so))
    miss = [r for r in rows if not r['hit']]; cnt = collections.Counter(r['cls'] for r in miss)
    tax = json.load(open(TAX))['routes']
    rep['taxonomy'] = dict(gold_entities=len(rows), hit=len(rows) - len(miss), missed=len(miss),
                           classes={c: dict(n=cnt[c], share=cnt[c] / len(miss) if miss else None) for c in CLASSES},
                           cd_overlap_stopwords_only=sum(r['overlap_stopwords_only'] for r in miss),
                           missed_by_type=collections.Counter(r['type'] for r in miss).most_common(),
                           e_by_type=collections.Counter(r['type'] for r in miss if r['cls'] == 'e_missed_extraction').most_common(),
                           examples={c: [dict(file=r['file'], gold=[r['type'], r['value']], pred=r['pred'], transcript=r['transcript'])
                                         for r in miss if r['cls'] == c][:10] for c in CLASSES},
                           side_by_side={rt: dict(missed=tax[rt]['all']['missed'], hit=tax[rt]['all']['hit'],
                                                  classes={c: tax[rt]['all']['classes'][c]['n'] for c in CLASSES}) for rt in ('P2', 'T2')})
    with open(out / 'entities_C0.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    # overlap with 22011390 T2
    old = {json.loads(l)['file']: json.loads(l) for l in open(OLD_T2)}
    com = sorted(set(old) & set(R))
    rep['overlap_22011390'] = dict(n=len(com), identical_prompt_text=sum(old[f]['prompt_text'] == R[f]['prompt_text'] for f in com),
                                   identical_raw_output=sum(old[f]['raw_output'] == R[f]['raw_output'] for f in com),
                                   differing=[f for f in com if old[f]['raw_output'] != R[f]['raw_output']])
    # pre-registered reading
    d = rep['binding_v1']['delta_C0_minus_C1P2']
    p2c = tax['P2']['all']['classes']; p2m = tax['P2']['all']['missed']
    bcd = (p2c['b_spelling']['n'] + p2c['c_boundary']['n'] + p2c['d_type']['n']) / p2m
    if d <= 0.05 + 1e-12 and bcd >= 0.5 - 1e-12:
        v = '主要是标注格式问题'
    elif d >= 0.15 - 1e-12:
        v = '音频路径有明显损失'
    else:
        v = '两者都有'
    rep['reading'] = dict(delta=d, p2_bcd_share=bcd, verdict=v)
    (out / 'eval_c0_v1.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: rep[k] for k in ('slu_v2', 'binding_v1', 'overlap_22011390', 'reading')}, ensure_ascii=False, indent=1))
    print('TAX', json.dumps({k: rep['taxonomy'][k] for k in ('gold_entities', 'hit', 'missed', 'classes', 'cd_overlap_stopwords_only', 'e_by_type')}, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:3])
