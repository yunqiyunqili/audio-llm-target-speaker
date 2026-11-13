"""Transcript diagnostics of P2 vs P0 from existing raw outputs only (no generation). Writes a NEW dir.
WER (slu-score-v2 normalize_v2, rescore_slu_v2.score_item) on 100 items; transcript-based binding diagnostic on the
76 tier-1 targets (binding_score_v1: transcript_counter / judge; per recording, then mean): target_any, only_target
(plus target_all, neither for completeness). Frozen code imported unchanged.
Usage: python -I scripts/qwen/p2_transcript_diag.py DATA_DIR P0_DIR P2_DIR OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from rescore_slu_v2 import score_item  # noqa: E402
from binding_score_v1 import judge, qualified_pairs, transcript_counter  # noqa: E402


def main(data, p0, p2, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _, pairs = qualified_pairs(data)
    partners = collections.defaultdict(list)
    for a, b, g, dd, Da, Db in pairs:
        if g and dd <= 1.0 and Da and Db:
            partners[a].append((Da, Db)); partners[b].append((Db, Da))
    targets = [it['file'] for it in json.load(open(PROJ / 'configs/c1_ext_devel100.json'))['items'] if it['tier'] == 1]
    rep = {}
    for name, d in (('P0', p0), ('P2', p2)):
        raw = {json.loads(l)['file']: json.loads(l) for l in open(Path(d) / 'raw.jsonl')}
        sc = {f: score_item(r) for f, r in raw.items()}
        we = sum(s['wer_err'] for s in sc.values()); wr = sum(s['wer_ref'] for s in sc.values())
        we1 = sum(s['wer_err_v1'] for s in sc.values())
        recs = []
        for f in targets:
            tr = sc[f]['hyp']['transcript']
            js = [judge(DT, DI, transcript_counter(tr, set(DT) | set(DI))) for DT, DI in partners[f]]
            n = len(js); c = collections.Counter(j['cat'] for j in js)
            recs.append(dict(target_any=sum(j['t_any'] for j in js) / n, target_all=sum(j['t_all'] for j in js) / n,
                             only_target=c['only_target'] / n, neither=c['neither'] / n,
                             only_interferer=c['only_interferer'] / n, both=c['both'] / n))
        rep[name] = dict(dir=str(d), wer=we / wr, wer_counts=[we, wr], wer_v1=we1 / wr,
                         items_with_transcript_error=sum(s['wer_err'] > 0 for s in sc.values()),
                         transcript_binding={k: sum(r[k] for r in recs) / len(recs) for k in recs[0]}, recordings=len(recs))
    (out / 'p2_transcript_diag.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
