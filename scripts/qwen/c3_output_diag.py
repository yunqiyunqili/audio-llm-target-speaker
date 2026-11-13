"""Read-only diagnostics of C3 outputs (job 22014089). Nothing is re-scored; frozen rules are not changed.
Part 1 - breakdown of the samples whose binding-score-v1 category is 'neither'. Categories are mutually exclusive and
  assigned in this order (rules fixed before looking at the counts):
    1. non_english          : the parsed transcript contains an alphabetic character outside ASCII (e.g. CJK), or the
                              output did not parse and its raw text contains such a character
    2. partial_entity       : >= 1 parsed entity value (normalize_v2, full value) equals a reference entity value of the
                              target or of the interferer (distinguishing or shared), but the binding category is 'neither'
    3. transcript_close     : no such entity value, but the normalised transcript has WER < 0.5 to the target sentence
                              or to the interferer sentence (entities not extracted although the words are close)
    4. unrelated            : none of the above (no matching entity value, WER >= 0.5 to both sentences)
  5 examples per category: random.Random(2026).sample over the category members sorted by key.
Part 2 - transcript choice: WER (normalize_v2) of the parsed transcript to target and to interferer. A sample is a
  'clear choice' only if min(WER) < 0.5 and |WER_t - WER_i| >= 0.1; it then chose the closer speaker. Reported overall
  and by full-image SIR stratum (< -0.5, [-0.5, 0.5], > 0.5 dB; manifest). Diagnostic only, not a gate conclusion.
Usage: python -I scripts/qwen/c3_output_diag.py RUN_DIR EVAL_DIR MIX_MANIFEST OUT_DIR
"""
import collections
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import parse, wer_counts  # noqa: E402

ORDER = ('non_english', 'partial_entity', 'transcript_close', 'unrelated')
SIR_BINS = (('<-0.5', -1e9, -0.5), ('[-0.5,0.5]', -0.5, 0.5000001), ('>0.5', 0.5000001, 1e9))


def non_ascii_alpha(s):
    return any(ord(c) > 127 and c.isalpha() for c in s)


def wer(ref, hyp):
    e, n = wer_counts(normalize_v2(ref), normalize_v2(hyp))
    return e / n


def main(run, ev, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    raw = {json.loads(l)['file']: json.loads(l) for l in open(Path(run) / 'C3/raw.jsonl')}
    cat = {json.loads(l)['key']: json.loads(l)['cat'] for l in open(Path(ev) / 'c3_per_sample.jsonl')}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    rows = []
    for k in sorted(raw):
        r = raw[k]; g = L[r['group']]; t = r['target']; i = 'B' if t == 'A' else 'A'
        obj, st = parse(r['raw_output'])
        tr = obj['transcript'] if obj else ''
        hv = {normalize_v2(v) for _, v in (obj['entities'] if obj else [])}
        refv = {normalize_v2(v) for _, v in g[t]['entities'] + g[i]['entities']}
        wt, wi = wer(g[t]['sentence'], tr), wer(g[i]['sentence'], tr)
        if non_ascii_alpha(tr if obj else r['raw_output']):
            c = 'non_english'
        elif hv & refv:
            c = 'partial_entity'
        elif min(wt, wi) < 0.5:
            c = 'transcript_close'
        else:
            c = 'unrelated'
        clear = min(wt, wi) < 0.5 and abs(wt - wi) >= 0.1
        rows.append(dict(key=k, binding=cat[k], neither_class=c if cat[k] == 'neither' else None, wer_target=wt, wer_interferer=wi,
                         clear_choice=('target' if wt < wi else 'interferer') if clear else None,
                         sir_full=man[k]['sir_mic0_full_db'], target_sentence=g[t]['sentence'], interferer_sentence=g[i]['sentence'],
                         target_entities=g[t]['entities'], interferer_entities=g[i]['entities'], raw_output=r['raw_output']))
    nei = [x for x in rows if x['binding'] == 'neither']
    groups = {c: [x for x in nei if x['neither_class'] == c] for c in ORDER}
    ex = {c: random.Random(2026).sample(v, min(5, len(v))) for c, v in groups.items()}
    def choice(sel):  # noqa: E306
        c = collections.Counter(x['clear_choice'] for x in sel)
        return dict(n=len(sel), clear=c['target'] + c['interferer'], target=c['target'], interferer=c['interferer'])
    rep = dict(run=str(run), neither_total=len(nei), neither_breakdown={c: len(v) for c, v in groups.items()},
               transcript_choice_all=choice(rows),
               transcript_choice_by_sir_full={b: choice([x for x in rows if lo <= x['sir_full'] < hi]) for b, lo, hi in SIR_BINS},
               transcript_choice_by_binding={b: choice([x for x in rows if x['binding'] == b]) for b in sorted({x['binding'] for x in rows})},
               examples={c: [{k: x[k] for k in ('key', 'target_sentence', 'interferer_sentence', 'target_entities', 'interferer_entities',
                                                 'wer_target', 'wer_interferer', 'raw_output')} for x in v] for c, v in ex.items()})
    (out / 'c3_output_diag.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    with open(out / 'c3_output_diag_per_sample.jsonl', 'w') as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'examples'}, indent=1))
    for c, v in rep['examples'].items():
        for x in v:
            print(c, '|', x['key'], '| T:', x['target_sentence'], '| I:', x['interferer_sentence'], '| wer', round(x['wer_target'], 2),
                  round(x['wer_interferer'], 2), '|', x['raw_output'][:200].replace('\n', ' '))


if __name__ == '__main__':
    main(*sys.argv[1:5])
