"""C0 vs C1/P2 intent diagnosis (descriptive only; reads existing raw outputs; no rule or prompt change).
Lists recordings where C0 intent is wrong and P2 right, and vice versa (strict intent as slu-score-v2 score_item int_ok),
with gold / predicted intents, reference sentence length, and P2's own transcript.
Usage: python -I -B scripts/qwen/c0_intent_diag.py OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from rescore_slu_v2 import score_item  # noqa: E402

O = PROJ / 'outputs'
C0 = O / 'qwen_omni_c0/22023448/C0/raw.jsonl'
P2 = O / 'qwen_omni_c3_p2/22014089/C1/raw.jsonl'


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    A, B = load(C0), load(P2)
    assert set(A) == set(B) and len(A) == 124
    SA = {f: score_item(r) for f, r in A.items()}; SB = {f: score_item(r) for f, r in B.items()}
    rows = []
    for f in sorted(A):
        assert A[f]['ref_intent'] == B[f]['ref_intent'] and A[f]['ref_sentence'] == B[f]['ref_sentence']
        rows.append(dict(file=f, gold=A[f]['ref_intent'], sentence=A[f]['ref_sentence'], n_words=len(A[f]['ref_sentence'].split()),
                         c0_intent=SA[f]['hyp']['intent'], c0_ok=SA[f]['int_ok'], c0_scen_ok=SA[f]['scen_ok'], c0_status=SA[f]['status'],
                         p2_intent=SB[f]['hyp']['intent'], p2_ok=SB[f]['int_ok'], p2_scen_ok=SB[f]['scen_ok'], p2_status=SB[f]['status'],
                         p2_transcript=SB[f]['hyp']['transcript']))
    grp = dict(c0_wrong_p2_right=[r for r in rows if not r['c0_ok'] and r['p2_ok']], p2_wrong_c0_right=[r for r in rows if not r['p2_ok'] and r['c0_ok']], both_wrong=[r for r in rows if not r['c0_ok'] and not r['p2_ok']],
               both_right=[r for r in rows if r['c0_ok'] and r['p2_ok']])
    nw = lambda rs: dict(mean=sum(r['n_words'] for r in rs) / len(rs) if rs else None, le2=sum(r['n_words'] <= 2 for r in rs))  # noqa: E731
    rep = dict(c0_intent_acc=sum(r['c0_ok'] for r in rows) / 124, p2_intent_acc=sum(r['p2_ok'] for r in rows) / 124,
               counts={k: len(v) for k, v in grp.items()},
               gold_intent_counts={k: collections.Counter(r['gold'] for r in v).most_common() for k, v in grp.items() if k != 'both_right'},
               gold_scenario_counts={k: collections.Counter(r['gold'].split('_')[0] for r in v).most_common() for k, v in grp.items() if k != 'both_right'},
               scenario_correct={k: sum(r['c0_scen_ok'] if k == 'c0_wrong_p2_right' else r['p2_scen_ok'] for r in v) for k, v in grp.items() if k in ('c0_wrong_p2_right', 'p2_wrong_c0_right')},
               n_words={k: nw(v) for k, v in grp.items()}, n_words_all=nw(rows),
               p2_transcript_has_punct_or_caps=sum(any(c in r['p2_transcript'] for c in '.,?!\'') or r['p2_transcript'] != r['p2_transcript'].lower() for r in rows),
               examples={k: v for k, v in grp.items() if k in ('c0_wrong_p2_right', 'p2_wrong_c0_right')})
    (out / 'c0_intent_diag.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for k in ('c0_wrong_p2_right', 'p2_wrong_c0_right'):
        print('==', k, len(grp[k]))
        for r in grp[k]:
            print(f"  {r['file'][:30]} gold={r['gold']} c0={r['c0_intent']} p2={r['p2_intent']} n={r['n_words']} | {r['sentence']} | P2: {r['p2_transcript']}")
    print(json.dumps({k: rep[k] for k in ('c0_intent_acc', 'p2_intent_acc', 'counts', 'gold_scenario_counts', 'scenario_correct', 'n_words', 'n_words_all', 'p2_transcript_has_punct_or_caps')}, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1])
