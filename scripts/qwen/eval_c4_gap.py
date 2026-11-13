"""C4 gap diagnostic (docs/c4_gap_diag_v1.md, frozen). Part A: FaSNet C4 'neither' samples by target-stream SI-SDR (direct).
Part B: C4 FaSNet vs C4-direct vs R1 with the same binding / M / U metrics (eval_c4_v1.score / summ imported unchanged).
Usage: python -I scripts/qwen/eval_c4_gap.py C4_RUN C4_DIRECT_RUN R1_RAW V2_PER_SAMPLE MIX_MANIFEST OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import score, summ  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import parse, wer_counts  # noqa: E402

SISDR_BINS = [(-1e9, 5), (5, 10), (10, 15), (15, 1e9)]


def wer(ref, hyp):
    e, n = wer_counts(normalize_v2(ref), normalize_v2(hyp)); return e / n


def main(c4, c4d, r1raw, v2ps, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    rawF = {json.loads(l)['file']: json.loads(l) for l in open(Path(c4) / 'fasnet_tac' / 'raw.jsonl')}
    rawD = {json.loads(l)['file']: json.loads(l) for l in open(Path(c4d) / 'C4_direct' / 'raw.jsonl')}
    r1 = {json.loads(l)['file']: json.loads(l) for l in open(r1raw)}
    rawR = {k: r1[L[m['group']][m['target']]['file']] for k, m in man.items()}
    R = dict(C4_fasnet_tac=score(rawF, L, man), C4_direct=score(rawD, L, man), R1=score(rawR, L, man, single=True))
    sis = {r['key']: r['ref']['direct']['sisdr_target'] for r in map(json.loads, open(v2ps)) if r['frontend'] == 'fasnet_tac'}
    # Part A
    A = {}
    rowsF = {r['key']: r for r in R['C4_fasnet_tac']}
    for lo, hi in SISDR_BINS:
        keys = [k for k in rowsF if lo <= sis[k] < hi]
        nei = [k for k in keys if rowsF[k]['cat'] == 'neither']
        ch = collections.Counter()
        for k in nei:
            m = man[k]; g = L[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'
            obj, _ = parse(rawF[k]['raw_output']); tr = obj['transcript'] if obj else ''
            wt, wi = wer(g[t]['sentence'], tr), wer(g[i]['sentence'], tr)
            clear = min(wt, wi) < 0.5 and abs(wt - wi) >= 0.1      # same definition as c3_output_diag.py
            ch[('target' if wt < wi else 'interferer') if clear else 'no_clear'] += 1
        A[f'[{lo:g},{hi:g})'] = dict(samples=len(keys), neither=len(nei), neither_frac=len(nei) / len(keys) if keys else None,
                                     M=sum(rowsF[k]['M'] for k in nei), U=sum(rowsF[k]['U'] for k in nei), clear_choice=dict(ch))
    rep = dict(spec='docs/c4_gap_diag_v1.md', neither_total=sum(r['cat'] == 'neither' for r in R['C4_fasnet_tac']), A_by_target_sisdr=A,
               B={c: summ(rows) for c, rows in R.items()})
    a = {c: v['target_any'] for c, v in rep['B'].items()}
    d_sep = a['C4_direct'] - a['C4_fasnet_tac']; d_chain = a['R1'] - a['C4_direct']
    verdict = []
    if d_sep >= 0.10 - 1e-12:
        verdict.append('separation residual and distortion are the main source of the C4 gap')
    if abs(d_chain) >= 0.10 - 1e-12:
        verdict.append(f"direct-path chain vs R1 chain difference matters ({'R1 higher' if d_chain > 0 else 'C4-direct higher'})")
    rep['decision'] = dict(d_sep=d_sep, d_chain=d_chain, verdict=verdict or ['undetermined'])
    (out / 'eval_c4_gap.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:7])
