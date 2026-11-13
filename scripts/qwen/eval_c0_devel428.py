"""Evaluate the C0 devel-428 baseline (docs/lora_slu_plan_v2.md). Reads raw outputs only. Descriptive.
- slu-score-v2 (eval_c3_v1.slu, unchanged); single-recording entity hit = any / all gold value hit (binding-score-v1 value matching,
  no interferer: D_T = all gold values, D_I empty); date recall; c1-error-taxonomy-v1 classes (functions unchanged, transcript =
  reference sentence); split into selection / report halves (plan §3.2); per-sentence duplicates; consistency with C0 v1 (job 22023448).
Usage: python -I -B scripts/qwen/eval_c0_devel428.py RAW OUT_DIR
"""
import collections
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from binding_score_v1 import judge, vals  # noqa: E402
from c1_error_taxonomy import CLASSES, contains, overlap, same  # noqa: E402
from eval_c3_v1 import slu  # noqa: E402
from rescore_slu_v2 import normalize_v2, score_item  # noqa: E402
from slu_normalize import parse  # noqa: E402

C0V1 = PROJ / 'outputs/qwen_omni_c0/22023448/C0/raw.jsonl'


def classify(gold, raw, trans):
    obj, st = parse(raw); pred = obj['entities'] if obj else []; left = vals(pred) if obj else collections.Counter(); out = []
    for t, v in gold:
        nv = normalize_v2(v)
        if left[nv] > 0:
            left[nv] -= 1; out.append((t, 'hit')); continue
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
        out.append((t, c))
    return out


def summary(R):
    S = {f: score_item(r) for f, r in R.items()}
    o = dict(n=len(R), slu_v2=slu(list(R.values()), S))
    js = {f: judge(vals(r['ref_entities']), collections.Counter(), vals(S[f]['hyp']['entities'])) for f, r in R.items()}
    o.update(any_gold_value_hit=sum(j['t_any'] for j in js.values()) / len(R), all_gold_values_hit=sum(j['t_all'] for j in js.values()) / len(R))
    cl = [x for f, r in R.items() for x in classify(r['ref_entities'], r['raw_output'], r['ref_sentence'])]
    o['gold_entities'] = len(cl); o['hit'] = sum(c == 'hit' for _, c in cl)
    o['classes'] = {c: sum(x == c for _, x in cl) for c in CLASSES}
    d = [c for t, c in cl if t == 'date']; o['date_gold'] = len(d); o['date_recall'] = sum(c == 'hit' for c in d) / len(d) if d else None
    o['date_missed_classes'] = dict(collections.Counter(c for c in d if c != 'hit'))
    return o


def main(raw, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    R = {json.loads(l)['file']: json.loads(l) for l in open(raw)}
    assert len(R) == 428
    rep = dict(spec='docs/lora_slu_plan_v2.md (C0 devel-428 baseline)', raw=str(raw), all=summary(R))
    ids = sorted({r['slurp_id'] for r in R.values()}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    H = dict(selection={f: r for f, r in R.items() if r['slurp_id'] in sel}, report={f: r for f, r in R.items() if r['slurp_id'] not in sel})
    rep['halves'] = {k: summary(v) for k, v in H.items()}
    rep['halves_sizes'] = {k: dict(recordings=len(v), sentences=len({r['slurp_id'] for r in v.values()})) for k, v in H.items()}
    by = collections.defaultdict(set)
    for r in R.values():
        by[r['slurp_id']].add(r['raw_output'])
    rep['duplicates'] = dict(sentences=len(by), sentences_with_multiple_recordings=sum(1 for i in by if sum(r['slurp_id'] == i for r in R.values()) > 1),
                             sentences_with_differing_outputs=sum(len(v) > 1 for v in by.values()))
    old = {json.loads(l)['file']: json.loads(l) for l in open(C0V1)}
    com = sorted(set(old) & set(R))
    rep['consistency_c0v1_22023448'] = dict(n=len(com), identical_prompt_text=sum(old[f]['prompt_text'] == R[f]['prompt_text'] for f in com),
                                            identical_raw_output=sum(old[f]['raw_output'] == R[f]['raw_output'] for f in com),
                                            differing=[f for f in com if old[f]['raw_output'] != R[f]['raw_output']])
    (out / 'eval_c0_devel428.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for k in ('all',):
        x = rep[k]; print(k, json.dumps({m: x[m] for m in ('n', 'any_gold_value_hit', 'all_gold_values_hit', 'gold_entities', 'hit', 'classes', 'date_gold', 'date_recall', 'date_missed_classes')}))
        print(' slu', json.dumps({m: round(v, 4) if isinstance(v, float) else v for m, v in x['slu_v2'].items()}))
    for k, x in rep['halves'].items():
        print(k, rep['halves_sizes'][k], json.dumps({m: x[m] for m in ('any_gold_value_hit', 'hit', 'gold_entities', 'date_recall', 'classes')}), 'intent', round(x['slu_v2']['intent_strict_acc'], 4))
    print('DUP', rep['duplicates']); print('CONSIST', {k: v for k, v in rep['consistency_c0v1_22023448'].items() if k != 'differing'}, rep['consistency_c0v1_22023448']['differing'][:5])


if __name__ == '__main__':
    main(*sys.argv[1:3])
