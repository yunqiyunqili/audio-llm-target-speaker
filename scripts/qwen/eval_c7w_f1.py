"""Evaluate F1 / R1-same (docs/fasnet_single_talker_v1.md) and C7-W (docs/c7_whisper_v1.md), rules frozen before running.
eval_c4_v1.score / summ unchanged. C1 sets are mapped to the 200 samples like R1 (target recording's output, sample's own pair).
Usage: python -I scripts/qwen/eval_c7w_f1.py RUN_DIR C5_RAW R1_RAW F1_STREAMS MIX_MANIFEST OUT_JSON
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import wer_counts  # noqa: E402

REF = dict(C5=0.395, C4direct_P2=0.670, C7_planA=0.255)


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(run, c5raw, r1raw, f1streams, manifest, out_json):
    run = Path(run)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    tf = {k: L[m['group']][m['target']]['file'] for k, m in man.items()}
    by_rec = lambda raw: {k: raw[tf[k]] for k in man}  # noqa: E731
    C = {}
    C['F1'] = score(load(run / 'F1' / 'raw.jsonl'), L, man, single=True)
    C['R1same'] = score(load(run / 'R1same' / 'raw.jsonl'), L, man, single=True)
    C['R1_original'] = score(by_rec(load(r1raw)), L, man, single=True)
    C['C5'] = score(load(c5raw), L, man)
    for s in ('C1', 'C4direct', 'C5'):
        raw = load(run / f'T_{s}' / 'raw.jsonl')
        C[f'C7W_{s}'] = score(by_rec(raw) if s == 'C1' else raw, L, man, single=(s != 'C5'))
    S = {c: summ(rows) for c, rows in C.items()}
    # Whisper WER
    W = {}
    for s in ('C1', 'C4direct', 'C5'):
        raw = load(run / f'W_{s}' / 'raw.jsonl'); raw = by_rec(raw) if s == 'C1' else raw
        et = ei = nt = ni = 0; lt05 = 0
        for k, m in man.items():
            g = L[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'; h = normalize_v2(raw[k]['raw_output'])
            a_, n_ = wer_counts(normalize_v2(g[t]['sentence']), h); b_, n2 = wer_counts(normalize_v2(g[i]['sentence']), h)
            et += a_; nt += n_; ei += b_; ni += n2; lt05 += a_ / n_ < 0.5
        W[s] = dict(wer_to_target=et / nt, wer_to_interferer=ei / ni, target_wer_lt_05=lt05, n=len(man))
    # 63-sample check: target RIR/position identical to original R1
    first = {}
    for g in sorted(L.values(), key=lambda g: g['group']):
        for k, pos in (('A', 0), ('B', 1)):
            first.setdefault(g[k]['file'], (g['group'], pos))
    same = [k for k, m in man.items() if first[tf[k]] == (m['group'], 0)]
    r1s = {r['key']: r for r in C['R1same']}; r1o = {r['key']: r for r in C['R1_original']}
    chk = dict(n=len(same), R1same=summ([r1s[k] for k in same]), R1_original=summ([r1o[k] for k in same]),
               identical_raw=sum(load(run / 'R1same' / 'raw.jsonl')[k]['raw_output'] == load(r1raw)[tf[k]]['raw_output'] for k in same))
    a = {c: v['target_any'] for c, v in S.items()}
    f1, r1, c5 = a['F1'], a['R1same'], a['C5']
    if abs(f1 - r1) <= 0.05 + 1e-12 and r1 - c5 >= 0.10 - 1e-12:
        f1v = 'residual-overlap-related problems more likely'
    elif r1 - f1 >= 0.10 - 1e-12 and abs(f1 - c5) <= 0.05 + 1e-12:
        f1v = 'FaSNet processing distortion more likely'
    else:
        f1v = 'in between or undetermined'
    w5, wd = a['C7W_C5'], a['C7W_C4direct']
    c7v = ('transcription + text route may suffice' if abs(w5 - wd) <= 0.05 + 1e-12 else
           'separated speech is also harder to transcribe' if wd - w5 >= 0.10 - 1e-12 else 'undetermined')
    f1s = [json.loads(l) for l in open(f1streams)]
    rep = dict(specs=['docs/fasnet_single_talker_v1.md', 'docs/c7_whisper_v1.md'], conditions=S, whisper=W, check_63=chk,
               f1_streams=dict(stream0=sum(r['stream'] == 0 for r in f1s), flipped=sum(r['sign'] < 0 for r in f1s),
                               median_sisdr_direct=sorted(r['sisdr_direct'] for r in f1s)[len(f1s) // 2],
                               median_rms_ratio=sorted(r['rms_ratio_low_high'] for r in f1s)[len(f1s) // 2]),
               decisions=dict(F1=dict(a_F1=f1, a_R1same=r1, a_C5=c5, verdict=f1v),
                              C7W=dict(a_C7W_C5=w5, a_C7W_C4direct=wd, verdict=c7v,
                                       reference_only={k: w5 - v for k, v in REF.items()})))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=1) + '\n')
    for c, v in S.items():
        print(c, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}))
    print('WHISPER', json.dumps(W)); print('CHECK63', json.dumps({k: (v if not isinstance(v, dict) else round(v['target_any'], 3)) for k, v in chk.items()}))
    print('F1STREAMS', json.dumps(rep['f1_streams'])); print('DECISIONS', json.dumps(rep['decisions'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:7])
