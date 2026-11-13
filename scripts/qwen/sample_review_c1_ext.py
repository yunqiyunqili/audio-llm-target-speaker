"""Review samples from the C1 extension (job 22010997), offline; reads the frozen v2 error files, writes to a NEW dir.
  * 20 items with error type entity_missing in T0 (RUN/T0/rescore_v2_errors.jsonl)
  * 10 items with error type transcript in P0 (RUN/P0/rescore_v2_errors.jsonl), with a word-level diff
Sampling: candidates sorted by file, random.Random(2026).sample(candidates, n). Labels are NOT changed.
Usage: python -I scripts/qwen/sample_review_c1_ext.py RUN_DIR OUT_DIR
"""
import difflib
import json
import random
import sys
from pathlib import Path

SEED = 2026


def pick(path, etype, n):
    c = sorted((json.loads(l) for l in open(path)), key=lambda e: e['file'])
    c = [e for e in c if etype in e['error_types']]
    return len(c), random.Random(SEED).sample(c, n)


def wdiff(a, b):
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False)
    return [dict(op=op, ref=' '.join(a.split()[i1:i2]), hyp=' '.join(b.split()[j1:j2]))
            for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']


def main(run, out):
    run, out = Path(run), Path(out); out.mkdir(parents=True, exist_ok=False)
    n1, t0 = pick(run / 'T0/rescore_v2_errors.jsonl', 'entity_missing', 20)
    n2, p0 = pick(run / 'P0/rescore_v2_errors.jsonl', 'transcript', 10)
    a = [dict(file=e['file'], slurp_id=e['slurp_id'], ref_sentence=e['ref_sentence'], ref_intent=e['ref_intent'],
              ref_entities=e['ref_entities'], hyp_intent=e['hyp_intent'], hyp_entities=e['hyp_entities'],
              error_types=e['error_types'], review_status=e['review_status']) for e in t0]
    b = [dict(file=e['file'], slurp_id=e['slurp_id'], ref=e['normalised_ref'], hyp=e['normalised_hyp'],
              diff=wdiff(e['normalised_ref'], e['normalised_hyp'])) for e in p0]
    (out / 't0_entity_missing_sample20.json').write_text(json.dumps(dict(seed=SEED, candidates=n1, items=a), indent=1, ensure_ascii=False) + '\n')
    (out / 'p0_transcript_sample10.json').write_text(json.dumps(dict(seed=SEED, candidates=n2, items=b), indent=1, ensure_ascii=False) + '\n')
    print('T0 entity_missing candidates', n1)
    for i, x in enumerate(a, 1):
        print(f'T{i:02d}', x['slurp_id'], '|', x['ref_sentence'], '| REF', x['ref_entities'], '| HYP', x['hyp_entities'],
              '|', x['ref_intent'], '->', x['hyp_intent'], x['review_status'])
    print('P0 transcript candidates', n2)
    for i, x in enumerate(b, 1):
        print(f'A{i:02d}', x['file'], '|', x['diff'])


if __name__ == '__main__':
    main(*sys.argv[1:3])
