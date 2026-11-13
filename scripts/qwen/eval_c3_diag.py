"""Evaluate the C3 diagnostic run (D1 / D2 / D3 on 40 mixtures). Offline; frozen scoring imported unchanged.
D3: raw output identical to job 22014089 C3 (per sample)?
All conditions: WER (normalize_v2) of the transcript to the target and to the interferer; 'clear choice' as in
  scripts/qwen/c3_output_diag.py (min WER < 0.5 and |diff| >= 0.1); non-English (non-ASCII alphabetic character).
  D1 transcript = the whole raw output (free text); also WER to the two sentences concatenated (both orders, min) ->
  'both speakers transcribed' if that WER < 0.5.
D2 / D3: parse status; binding-score-v1 category w.r.t. the theta1 target (s1 = A, s2 = B) and 'who' (only_A / only_B /
  both / neither, independent of the direction); unrelated = no reference value and WER >= 0.5 to both sentences.
Usage: python -I scripts/qwen/eval_c3_diag.py DIAG_RUN C3_RUN OUT_DIR
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
from c3_output_diag import non_ascii_alpha  # noqa: E402


def wer(ref, hyp):
    e, n = wer_counts(normalize_v2(ref), normalize_v2(hyp)); return e / n


def main(diag, c3, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    old = {json.loads(l)['file']: json.loads(l) for l in open(Path(c3) / 'C3/raw.jsonl')}
    rep, per = {}, []
    for cond in ('D1', 'D2', 'D3'):
        rows = [json.loads(l) for l in open(Path(diag) / cond / 'raw.jsonl')]
        agg = collections.Counter()
        for r in rows:
            g = L[r['group']]; t = r['target']; i = 'B' if t == 'A' else 'A'
            obj, st = parse(r['raw_output']) if cond != 'D1' else (None, 'free_text')
            tr = r['raw_output'] if cond == 'D1' else (obj['transcript'] if obj else '')
            wt, wi = wer(g[t]['sentence'], tr), wer(g[i]['sentence'], tr)
            both = min(wer(g[t]['sentence'] + ' ' + g[i]['sentence'], tr), wer(g[i]['sentence'] + ' ' + g[t]['sentence'], tr))
            clear = ('target' if wt < wi else 'interferer') if min(wt, wi) < 0.5 and abs(wt - wi) >= 0.1 else None
            x = dict(cond=cond, key=r['file'], wer_target=wt, wer_interferer=wi, wer_both=both, clear=clear,
                     non_english=non_ascii_alpha(tr or r['raw_output']), parse=st, raw=r['raw_output'])
            agg.update(n=1, non_english=x['non_english'], clear_target=clear == 'target', clear_interferer=clear == 'interferer',
                       both_transcribed=both < 0.5, any_wer_lt_05=min(wt, wi) < 0.5)
            if cond != 'D1':
                H = vals(obj['entities']) if obj else collections.Counter()
                VT, VI = vals(g[t]['entities']), vals(g[i]['entities'])
                VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
                x['cat'] = judge(VT - VI, VI - VT, H)['cat']
                x['who'] = dict(only_target='only_A', only_interferer='only_B', both='both', neither='neither')[judge(VA - VB, VB - VA, H)['cat']]
                refv = {normalize_v2(v) for _, v in g[t]['entities'] + g[i]['entities']}
                x['unrelated'] = not (set(H) & refv) and min(wt, wi) >= 0.5
                agg.update(parse_fail=st != 'ok', unrelated=x['unrelated'], **{f"cat_{x['cat']}": 1})
            if cond == 'D3':
                x['identical_to_22014089'] = r['raw_output'] == old[r['file']]['raw_output']
                agg.update(identical_to_22014089=x['identical_to_22014089'])
            per.append(x)
        rep[cond] = dict(agg)
    by = collections.defaultdict(dict)
    for x in per:
        by[x['key']][x['cond']] = x
    rep['D2_vs_D3'] = dict(same_who=sum(v['D2']['who'] == v['D3']['who'] for v in by.values()),
                           same_raw=sum(v['D2']['raw'] == v['D3']['raw'] for v in by.values()),
                           same_clear_choice=sum(v['D2']['clear'] == v['D3']['clear'] for v in by.values()))
    rep['swap_pairs_D2_identical_raw'] = sum(by[f'g{gi:03d}_s1']['D2']['raw'] == by[f'g{gi:03d}_s2']['D2']['raw']
                                             for gi in {x['key'][:4] and int(x['key'][1:4]) for x in per})
    (out / 'eval_c3_diag.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(out / 'per_sample.jsonl', 'w') as fh:
        for x in per:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
