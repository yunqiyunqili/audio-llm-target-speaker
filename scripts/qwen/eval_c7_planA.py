"""C7 plan A evaluation (docs/c7_planA_v1.md, frozen): C7T2 vs C5 vs C4-direct with eval_c4_v1.score / summ unchanged,
plus ASR (C7ASR) WER to target and interferer sentences (normalize_v2). Descriptive only.
Usage: python -I scripts/qwen/eval_c7_planA.py C6C7_RUN C5_RAW C4_DIRECT_RAW MIX_MANIFEST OUT_JSON
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import wer_counts  # noqa: E402


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(run, c5raw, c4draw, manifest, out_json):
    run = Path(run)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    asr = load(run / 'C7ASR' / 'raw.jsonl')
    R = dict(C7_planA=summ(score(load(run / 'C7T2' / 'raw.jsonl'), L, man)), C5=summ(score(load(c5raw), L, man)),
             C4_direct=summ(score(load(c4draw), L, man)))
    et = ei = nt = ni = 0; closer_t = closer_i = 0
    for k, m in man.items():
        g = L[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'; h = normalize_v2(asr[k]['raw_output'])
        a, n = wer_counts(normalize_v2(g[t]['sentence']), h); b, n2 = wer_counts(normalize_v2(g[i]['sentence']), h)
        et += a; nt += n; ei += b; ni += n2; closer_t += a / n < b / n2; closer_i += b / n2 < a / n
    rep = dict(spec='docs/c7_planA_v1.md', note='C7 input = C5 target stream (oracle query azimuth); descriptive only', conditions=R,
               asr=dict(wer_to_target=et / nt, wer_to_interferer=ei / ni, closer_to_target=closer_t, closer_to_interferer=closer_i, n=len(man)))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1) + '\n')
    for c, v in R.items():
        print(c, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}))
    print('ASR', json.dumps(rep['asr']))


if __name__ == '__main__':
    main(*sys.argv[1:6])
