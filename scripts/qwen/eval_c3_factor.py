"""Evaluate the C3 factor diagnostic per docs/c3_factor_diag_v1.md (rules frozen before the job). Offline; NEW out dir.
Frozen code imported unchanged (rescore_slu_v2.normalize_v2, slu_normalize.parse / wer_counts, binding_score_v1.vals / judge,
postproc_c1_ext.REFUSAL, c3_output_diag.non_ascii_alpha).
Usage: python -I scripts/qwen/eval_c3_factor.py FACTOR_RUN C1_RUN_DIR D2_RUN_DIR OUT_DIR
  C1_RUN_DIR = outputs/qwen_omni_c3_p2/22014089/C1 ; D2_RUN_DIR = outputs/qwen_omni_c3_diag/22014131/D2
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
from c3_output_diag import non_ascii_alpha  # noqa: E402

DELTA = 0.15


def wer(ref, hyp):
    e, n = wer_counts(normalize_v2(ref), normalize_v2(hyp)); return e / n


def common(r):
    obj, st = parse(r['raw_output'])
    tr = obj['transcript'] if obj else ''
    H = vals(obj['entities']) if obj else collections.Counter()
    return obj, st, tr, H, dict(non_english=non_ascii_alpha(tr if obj else r['raw_output']), parse_fail=st != 'ok',
                                refusal=st == 'no_json' and bool(REFUSAL.search(r['raw_output'])))


def eval_single(path, L):
    rows = [json.loads(l) for l in open(path)]
    opp = collections.defaultdict(list)
    for g in L['groups']:
        VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
        opp[g['A']['file']].append((VA - VB, VB - VA)); opp[g['B']['file']].append((VB - VA, VA - VB))
    per = {}; agg = collections.Counter()
    for r in rows:
        obj, st, tr, H, flags = common(r)
        js = [judge(DT, DI, H) for DT, DI in opp[r['file']]]; n = len(js)
        w = wer(r['ref_sentence'], tr)
        refv = {normalize_v2(v) for _, v in r['ref_entities']}
        x = dict(target_any=sum(j['t_any'] for j in js) / n, only_target=sum(j['cat'] == 'only_target' for j in js) / n,
                 neither=sum(j['cat'] == 'neither' for j in js) / n, M=w < 0.5, U=not (set(H) & refv) and w >= 0.5, **flags)
        per[r['file']] = x
        agg.update(M=x['M'], U=x['U'], **{k: flags[k] for k in flags})
    n = len(rows)
    return dict(n=n, target_any=sum(x['target_any'] for x in per.values()) / n, only_target=sum(x['only_target'] for x in per.values()) / n,
                neither=sum(x['neither'] for x in per.values()) / n, M=agg['M'], U=agg['U'], non_english=agg['non_english'],
                refusal=agg['refusal'], parse_fail=agg['parse_fail'], m=agg['M'] / n, u=agg['U'] / n), per


def eval_mixed(path, L):
    rows = [json.loads(l) for l in open(path)]
    gl = {g['group']: g for g in L['groups']}
    agg = collections.Counter(); per = {}
    for r in rows:
        g = gl[r['group']]; t = r['target']; i = 'B' if t == 'A' else 'A'
        obj, st, tr, H, flags = common(r)
        VT, VI = vals(g[t]['entities']), vals(g[i]['entities'])
        j = judge(VT - VI, VI - VT, H)
        wt, wi = wer(g[t]['sentence'], tr), wer(g[i]['sentence'], tr)
        refv = {normalize_v2(v) for _, v in g[t]['entities'] + g[i]['entities']}
        x = dict(cat=j['cat'], t_any=j['t_any'], M=min(wt, wi) < 0.5, U=not (set(H) & refv) and min(wt, wi) >= 0.5, **flags)
        per[r['file']] = x
        agg.update(t_any=x['t_any'], only_target=x['cat'] == 'only_target', only_interferer=x['cat'] == 'only_interferer',
                   neither=x['cat'] == 'neither', M=x['M'], U=x['U'], **flags)
    n = len(rows)
    return dict(n=n, target_any=agg['t_any'] / n, only_target=agg['only_target'] / n, only_interferer=agg['only_interferer'] / n,
                neither=agg['neither'] / n, M=agg['M'], U=agg['U'], non_english=agg['non_english'], refusal=agg['refusal'],
                parse_fail=agg['parse_fail'], m=agg['M'] / n, u=agg['U'] / n), per


def main(run, c1dir, d2dir, out):
    run, out = Path(run), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))
    R = {}
    R['C1'], c1per = eval_single(Path(c1dir) / 'raw.jsonl', L)
    for c in ('C1b', 'C2', 'R1'):
        R[c], _ = eval_single(run / c / 'raw.jsonl', L)
    R['D2'], d2per = eval_mixed(Path(d2dir) / 'raw.jsonl', L)
    for c in ('X1', 'X2', 'X3'):
        R[c], _ = eval_mixed(run / c / 'raw.jsonl', L)
    # C1 restricted to the targets of the 40 mixed samples
    gl = {g['group']: g for g in L['groups']}
    tg = [gl[int(k[1:4])]['A' if k.endswith('s1') else 'B']['file'] for k in d2per]
    R['C1_40'] = dict(n=len(tg), M=sum(c1per[f]['M'] for f in tg), m=sum(c1per[f]['M'] for f in tg) / len(tg))
    m = {k: v['m'] for k, v in R.items()}; u = {k: v.get('u') for k, v in R.items()}
    rules = [('bandwidth_single', 'C1', 'C1b'), ('early_reverb_single', 'C1b', 'C2'), ('late_reverb_single', 'C2', 'R1'),
             ('overlap_dry', 'C1_40', 'X1'), ('bandwidth_overlap', 'X1', 'X2'), ('reverb_noise_overlap', 'X2', 'D2')]
    dec = []
    for name, a, b in rules:
        drop = m[a] - m[b]
        dec.append(dict(factor=name, compare=f'{a} -> {b}', m_from=m[a], m_to=m[b], drop=drop, effect=drop >= DELTA - 1e-12, size=drop))
    dm, du = m['X3'] - m['D2'], u['X3'] - u['D2']
    dec.append(dict(factor='P2_constraint', compare='D2 <-> X3 (P0)', dm=dm, du=du,
                    effect=abs(dm) >= DELTA - 1e-12 or abs(du) >= DELTA - 1e-12, size=max(abs(dm), abs(du))))
    eff = [d for d in dec if d['effect']]
    main_src = max(eff, key=lambda d: d['size'])['factor'] if eff else 'undetermined (no factor meets the threshold)'
    rep = dict(spec='docs/c3_factor_diag_v1.md', delta=DELTA, conditions=R, decisions=dec, main_source=main_src,
               vs_D2={c: {k: R[c][k] - R['D2'][k] for k in ('target_any', 'only_target', 'neither', 'M', 'U', 'non_english', 'refusal')}
                      for c in ('X1', 'X2', 'X3')})
    (out / 'eval_c3_factor.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
