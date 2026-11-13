"""Evaluate C5 v1 (docs/c5_v1.md, frozen) against C3, C4 FaSNet-TAC, C4-direct, R1 with identical metrics and samples
(eval_c4_v1.score / summ imported unchanged). Selection correctness uses v2 ref_stream_direct (evaluation only).
Usage: python -I scripts/qwen/eval_c5_v1.py C5_RUN C4_RUN C4_DIRECT_RUN C3_RAW R1_RAW V2_PER_SAMPLE SELECTION MIX_MANIFEST OUT_DIR
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from eval_c4_v1 import BINS, score, summ  # noqa: E402


def load(p):
    return {json.loads(l)['file']: json.loads(l) for l in open(p)}


def main(c5, c4, c4d, c3raw, r1raw, v2ps, selp, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(manifest))}
    raw5 = load(Path(c5) / 'C5_fasnet_tac' / 'raw.jsonl'); raw4 = load(Path(c4) / 'fasnet_tac' / 'raw.jsonl')
    r1 = load(r1raw)
    R = dict(C5_fasnet_tac=score(raw5, L, man), C4_fasnet_tac=score(raw4, L, man),
             C4_direct=score(load(Path(c4d) / 'C4_direct' / 'raw.jsonl'), L, man), C3=score(load(c3raw), L, man),
             R1=score({k: r1[L[m['group']][m['target']]['file']] for k, m in man.items()}, L, man, single=True))
    ref = {r['key']: r['ref_stream_direct'] for r in map(json.loads, open(v2ps)) if r['frontend'] == 'fasnet_tac'}
    sel = {r['key']: r for r in map(json.loads, open(selp))}
    rep = dict(spec='docs/c5_v1.md', oracle_note='C5 uses the ground-truth target azimuth as the query; not fully deployable', conditions={})
    for c, rows in R.items():
        st = {}
        for k, bins in BINS.items():
            st[k] = ({s: summ([r for r in rows if r['stratum'] == s]) for s in sorted({r['stratum'] for r in rows})} if bins is None else
                     {f'[{lo:g},{hi:g})': summ([r for r in rows if lo <= r[k] < hi]) for lo, hi in bins})
        rep['conditions'][c] = dict(all=summ(rows), strata=st)
    rows5 = {r['key']: r for r in R['C5_fasnet_tac']}
    correct = [k for k in rows5 if sel[k]['stream'] == ref[k]]; wrong = [k for k in rows5 if sel[k]['stream'] != ref[k]]
    rep['selection'] = dict(correct=len(correct), wrong=len(wrong), wrong_keys=wrong,
                            correct_excl_0_15=sum(1 for k in correct if rows5[k]['stratum'] != '[0,15)'),
                            n_excl_0_15=sum(1 for k in rows5 if rows5[k]['stratum'] != '[0,15)'),
                            same_stream_as_C4=sum(sel[k]['stream'] == raw4[k]['stream'] for k in rows5),
                            identical_raw_to_C4_when_same_stream=sum(raw5[k]['raw_output'] == raw4[k]['raw_output'] for k in rows5 if sel[k]['stream'] == raw4[k]['stream']))
    rep['by_selection'] = dict(correct=summ([rows5[k] for k in correct]),
                               wrong=(summ([rows5[k] for k in wrong]) if len(wrong) >= 5 else
                                      [dict(key=k, cat=rows5[k]['cat'], t_any=rows5[k]['t_any'], stratum=rows5[k]['stratum']) for k in wrong]))
    fails = [k for k in correct if rows5[k]['cat'] in ('neither', 'only_interferer')]
    rep['selection_correct_but_downstream_failed'] = dict(n=len(fails), neither=sum(rows5[k]['cat'] == 'neither' for k in fails),
                                                          only_interferer=sum(rows5[k]['cat'] == 'only_interferer' for k in fails),
                                                          keys=fails, M=sum(rows5[k]['M'] for k in fails), U=sum(rows5[k]['U'] for k in fails))
    a = {c: v['all']['target_any'] for c, v in rep['conditions'].items()}
    rep['judgements'] = dict(
        c5_vs_c4=dict(delta=a['C5_fasnet_tac'] - a['C4_fasnet_tac'], equivalent=abs(a['C5_fasnet_tac'] - a['C4_fasnet_tac']) <= 0.05 + 1e-12),
        c5_vs_c3=dict(delta=a['C5_fasnet_tac'] - a['C3'], helps_given_true_azimuth=a['C5_fasnet_tac'] - a['C3'] >= 0.15 - 1e-12),
        c4direct_vs_c5=dict(delta=a['C4_direct'] - a['C5_fasnet_tac'], separation_degradation_still_bottleneck=a['C4_direct'] - a['C5_fasnet_tac'] >= 0.10 - 1e-12))
    (out / 'eval_c5_v1.json').write_text(json.dumps(rep, indent=1) + '\n')
    for c, v in rep['conditions'].items():
        print(c, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v['all'].items()}))
    print(json.dumps({k: rep[k] for k in ('selection', 'judgements')}, indent=1))
    print('FAILED_BUT_CORRECT', json.dumps({k: v for k, v in rep['selection_correct_but_downstream_failed'].items() if k != 'keys'}))


if __name__ == '__main__':
    main(*sys.argv[1:10])
