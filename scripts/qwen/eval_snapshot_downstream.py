"""Downstream decision for snapshot front-ends (docs/frontend_compare_v1.md; thresholds unchanged: P2 >= 0.495, Whisper+T2 >= 0.440).
eval_c4_v1.score / summ unchanged; Whisper WER (normalize_v2). IN-TRAINING snapshots, diagnosis only.
Usage: python -I scripts/qwen/eval_snapshot_downstream.py OUT_JSON model=RUN_DIR ...
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import wer_counts  # noqa: E402
import split_cfg  # noqa: E402  (2026-11-12 SLURP test switch; devel default unchanged)

THR = dict(P2=0.495, T2=0.440)
FASNET = dict(P2=0.395, T2=0.340)


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(out_json, *specs):
    L = {g['group']: g for g in json.load(open(split_cfg.LIST))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(split_cfg.MANIFEST))}
    rep = dict(note='IN-TRAINING SNAPSHOTS, diagnosis only; not finished models; no test used', thresholds=THR, fasnet_reference=FASNET, models={})
    for spec in specs:
        m, run = spec.split('='); run = Path(run)
        S = {c: summ(score(load(run / c / 'raw.jsonl'), L, man)) for c in ('P2', 'T2')}
        W = load(run / 'W' / 'raw.jsonl'); et = nt = lt = 0
        for k, mm in man.items():
            g = L[mm['group']]; t = mm['target']; a_, n_ = wer_counts(normalize_v2(g[t]['sentence']), normalize_v2(W[k]['raw_output']))
            et += a_; nt += n_; lt += a_ / n_ < 0.5
        dec = {c: dict(a=S[c]['target_any'], threshold=THR[c], delta_vs_fasnet=S[c]['target_any'] - FASNET[c],
                       verdict='stronger front-end clearly helps' if S[c]['target_any'] >= THR[c] - 1e-12 else 'cannot confirm a clear benefit') for c in S}
        rep['models'][m] = dict(run=str(run), P2=S['P2'], WhisperT2=S['T2'], whisper=dict(wer_to_target=et / nt, target_wer_lt_05=lt), decisions=dec)
        print(m, 'P2', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S['P2'].items()}))
        print(m, 'W+T2', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S['T2'].items()}), 'WER', round(et / nt, 4), lt)
        print(m, 'DECISIONS', json.dumps(dec))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True); Path(out_json).write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
