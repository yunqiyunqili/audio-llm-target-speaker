"""SLURP test v1 one-shot scoring (docs/slurp_test_freeze_v1.md §0, §5, §6). CPU only; reads raw outputs; no model.

Primary comparisons (Audio LoRA - Whisper+T2-LoRA, target_any, paired; Bonferroni over 3):
  C1, C1-R : compare_audio_vs_t2.hit / boot rule (binding-score-v1 target_any with empty interferer set; bootstrap grouped by
             slurp_id, B=2000, default_rng(2026)); the same draws give the 95% and 98.33% (0.8333%/99.1667%) intervals.
  C5       : eval_final_lora_checks.py --route-kind audio_vs_t2 (stratified group bootstrap), C5 / none / all / target_any / eq
             (raw equal-sample metric, decision 2026-11-11); ci95 and ci9833 from that file.
  Significant <=> 98.33% lower bound > 0. Everything else is descriptive.
Descriptive: eval_c0_devel428.summary per single route; Whisper WER (eval_baselines428.wer); C3 with eval_step1_checks.mix_rows /
  msum / ents (the frozen §6 C3 scoring) for base P2 and Audio LoRA; C5 / c6-real-v1 / risk checks read from the frozen scoring
  outputs (eval_final_routes.py, eval_final_lora_checks.py), cv_dev93 natural weights (main) and test_eval92 (descriptive).
Imported functions are unchanged. --devel-check reruns the single-condition and C3 pieces on existing devel outputs and asserts the
recorded devel numbers are reproduced exactly (S0 equivalence check).
Usage: python -I -B scripts/lora/score_slurp_test_v1.py --inputs INPUTS.json --out DIR
       python -I -B scripts/lora/score_slurp_test_v1.py --devel-check OUT_JSON
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
for d in ('scripts/qwen', 'scripts/slurp', 'scripts/lora'):
    sys.path.insert(0, str(PROJ / d))
from compare_audio_vs_t2 import boot, hit  # noqa: E402
from eval_c0_devel428 import summary  # noqa: E402
from eval_baselines428 import wer  # noqa: E402
from eval_step1_checks import ents, mix_rows, msum  # noqa: E402

Q9833 = [0.05 / 3 / 2, 1 - 0.05 / 3 / 2]
KEYS = ('any_gold_value_hit', 'all_gold_values_hit', 'hit', 'gold_entities', 'date_gold', 'date_recall', 'classes')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(paths, n=None):
    R = {}
    for p in paths:
        for line in open(PROJ / p):
            r = json.loads(line); assert r['file'] not in R, (p, r['file']); R[r['file']] = r
    assert n is None or len(R) == n, (paths, len(R), n)
    return R


def pboot(A, B):
    """compare_audio_vs_t2 rule (A - B), plus 98.33% from the same draws; asserts identity with boot()."""
    assert set(A) == set(B)
    for k in A:
        assert A[k]['slurp_id'] == B[k]['slurp_id'] and A[k]['ref_entities'] == B[k]['ref_entities'], k
    d = {k: (A[k]['slurp_id'], hit(A[k]) - hit(B[k])) for k in A}
    ref = boot(d)
    by = collections.defaultdict(list)
    for k, v in d.items():
        by[v[0]].append(v[1])
    s = np.array([sum(v) for v in by.values()]); c = np.array([len(v) for v in by.values()])
    ix = np.random.default_rng(2026).integers(0, len(s), (2000, len(s))); z = s[ix].sum(1) / c[ix].sum(1)
    ci95 = [float(x) for x in np.quantile(z, [.025, .975])]; ci9833 = [float(x) for x in np.quantile(z, Q9833)]
    assert ref['delta'] == float(s.sum() / c.sum()) and ref['ci95'] == ci95
    return dict(n=len(A), n_slurp_id=len(by), a_any=sum(hit(x) for x in A.values()) / len(A), b_any=sum(hit(x) for x in B.values()) / len(B),
                delta=ref['delta'], ci95=ci95, ci9833=ci9833, B=2000, seed=2026)


def desc(R):
    s = summary(R)
    return dict({k: s[k] for k in KEYS}, n=s['n'], intent_strict_acc=s['slu_v2']['intent_strict_acc'], parse_failures=s['slu_v2']['parse_failures'],
                zero_entity=sum(1 for x in R.values() if not hyp_entities(x)), slu_v2=s['slu_v2'])


def hyp_entities(r):
    from rescore_slu_v2 import score_item
    return score_item(r)['hyp']['entities']


def devel_check(out):
    O = 'outputs/'
    rep = {}
    # (a) primary single-condition rule on devel all = comparison.json C1_all / C1R_all
    import compare_audio_vs_t2_halves as H
    cmp = json.load(open(PROJ / 'outputs/lora_step2_eval/audio_vs_t2_comparison_v2/comparison.json'))
    for cond in ('C1', 'C1R'):
        A = {}; T = {}
        for h in ('selection', 'report'):
            a, t = H.load(cond, h); A.update(a); T.update(t)
        got = pboot(A, T); exp = cmp[f'{cond}_all']
        assert got['delta'] == exp['paired_audio_minus_t2']['delta'] and got['ci95'] == exp['paired_audio_minus_t2']['ci95'], cond
        assert got['a_any'] == exp['audio_any'] and got['b_any'] == exp['t2_any'] and got['n'] == exp['n'], cond
        rep[f'{cond}_all'] = dict(reproduced=True, delta=got['delta'], ci95=got['ci95'], ci9833_new=got['ci9833'])
    # (b) descriptive summary + WER = eval_baselines428.json (all)
    import glob
    b428 = json.load(open(PROJ / 'outputs/lora_base_audio_eval/v1/eval_baselines428.json'))
    for k, pat in (('P2', 'lora_base_audio/p2_devel_clean/*/P2'), ('T', 'lora_base_audio/whisper_t2_devel_clean/*/T'),
                   ('P2_R', 'lora_base_audio/p2_devel_rev/*/P2'), ('T_R', 'lora_base_audio/whisper_t2_devel_rev/*/T')):
        R = load(sorted(str(Path(p).relative_to(PROJ)) for p in glob.glob(str(PROJ / O / pat / 'raw.jsonl'))), 428)
        d = desc(R); e = b428['conditions'][k]['all']
        for m in ('any_gold_value_hit', 'hit', 'gold_entities', 'date_recall', 'classes', 'intent_strict_acc', 'parse_failures'):
            assert d[m] == e[m], (k, m, d[m], e[m])
        rep[f'summary_{k}'] = 'reproduced'
    for k, pat in (('W', 'lora_base_audio/whisper_t2_devel_clean/*/W'), ('W_R', 'lora_base_audio/whisper_t2_devel_rev/*/W')):
        R = load(sorted(str(Path(p).relative_to(PROJ)) for p in glob.glob(str(PROJ / O / pat / 'raw.jsonl'))), 428)
        assert wer(R) == b428['wer'][k]['all'], k
        rep[f'wer_{k}'] = 'reproduced'
    # (c) C3 frozen §6 scoring = eval_step1_checks.json mix c3 all
    s1 = json.load(open(PROJ / 'outputs/lora_step1_checks/nonspatial_20261107/eval_step1_checks.json'))
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']
    lp = s1['mix']['c3']['lora_raw']; bp = s1['mix']['c3']['baseline_raw']
    lr = mix_rows(load([str(Path(lp).relative_to(PROJ)) if Path(lp).is_absolute() else lp], 200), L)
    br = mix_rows(load([str(Path(bp).relative_to(PROJ)) if Path(bp).is_absolute() else bp], 200), L)
    assert msum(list(lr.values())) == s1['mix']['c3']['all']['lora'] and msum(list(br.values())) == s1['mix']['c3']['all']['p2']
    assert ents(list(lr.values())) == s1['mix']['c3']['entity_counts_all_samples']['lora']
    rep['C3_all'] = 'reproduced'
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1)); print('DEVEL_CHECK_PASS', flush=True)


def main(inputs, out):
    out = Path(out); assert not out.exists()
    I = json.load(open(inputs))
    nrec, nmix = I['n_records'], I['n_mix_samples']  # test: 1183 recordings; mix samples from the frozen test list
    rep = dict(spec='docs/slurp_test_freeze_v1.md', inputs=str(inputs), inputs_sha256=sha(inputs), primary={}, single={}, mix={}, other_comparisons={})
    S = {c: {r: load(p, nrec) for r, p in I['single'][c].items()} for c in I['single']}
    for c in ('C1', 'C1R'):
        files = set(S[c]['P2'])
        assert all(set(v) == files for v in S[c].values())
        # T (Whisper+T2) and T2-LoRA must see the same Whisper transcript (W)
        assert all(S[c]['T'][f]['asr_text'] == S[c]['W'][f]['raw_output'] == S[c]['T2LORA'][f]['asr_text'] for f in files), c
        rep['single'][c] = {r: desc(R) for r, R in S[c].items() if r != 'W'}
        rep['single'][c]['whisper_wer'] = wer(S[c]['W'])
        p = pboot(S[c]['AUDIO'], S[c]['T2LORA']); p['significant_bonferroni'] = p['ci9833'][0] > 0; p['lower95_gt_0'] = p['ci95'][0] > 0
        rep['primary'][c] = p
        rep['other_comparisons'][c] = dict(audio_minus_p2=pboot(S[c]['AUDIO'], S[c]['P2']), t2lora_minus_wt2=pboot(S[c]['T2LORA'], S[c]['T']))
    assert set(S['C0']['T2']) == set(S['C0']['T2LORA']) == set(S['C1']['P2'])
    rep['single']['C0'] = {r: desc(R) for r, R in S['C0'].items()}
    rep['single']['C0']['by_sentence_note'] = 'C0 input is the sentence; recordings of the same sentence have identical inputs'
    rep['single']['C0']['sentences_with_differing_outputs'] = {r: sum(len(v) > 1 for v in _by_sid(R).values()) for r, R in S['C0'].items()}
    # C3 (frozen §6 scoring; risk check is descriptive on test)
    L = json.load(open(PROJ / I['mix_list']))['groups']
    C3 = {r: mix_rows(load(p, nmix), L) for r, p in I['c3'].items()}
    lr, br = list(C3['AUDIO'].values()), list(C3['P2'].values())
    rep['mix']['C3'] = dict(audio_lora=msum(lr), p2=msum(br), entity_counts=dict(audio_lora=ents(lr), p2=ents(br)))
    a, b = rep['mix']['C3']['audio_lora'], rep['mix']['C3']['p2']
    rep['mix']['C3']['devel_section6_rule_descriptive'] = dict(delta_only_interferer=a['only_interferer'] - b['only_interferer'], delta_target_any=a['target_any'] - b['target_any'],
                                                               would_pass=(a['only_interferer'] <= b['only_interferer'] + 0.02 + 1e-12) and (a['target_any'] >= b['target_any'] - 0.03 - 1e-12))
    # C5 / c6-real / paired checks from the frozen scoring outputs
    J = {k: json.load(open(PROJ / p)) for k, p in I['eval_json'].items()}
    for k in J:
        rep.setdefault('eval_json_sha256', {})[k] = sha(PROJ / I['eval_json'][k])
    c5 = J['checks_audio_vs_t2']['conditions']['C5']['none']['all']
    pt = c5['paired']['target_any']['eq']
    rep['primary']['C5'] = dict(n=c5['answered_observations'], a_any=c5['models']['p2']['target_any_eq'], b_any=c5['models']['lora']['target_any_eq'],
                                delta=pt['delta'], ci95=pt['ci95'], ci9833=pt['ci9833'], significant_bonferroni=pt['ci9833'][0] > 0, lower95_gt_0=pt['ci95'][0] > 0,
                                natural_weighted_descriptive=c5['paired']['target_any']['nat'], note='Audio LoRA (p2 slot) - Whisper+T2-LoRA (lora slot)')
    rep['mix']['C5'] = dict(base=J['c5_base']['models']['spatialnet'], audio_lora=J['c5_audio']['models']['spatialnet']['P2'],
                            audio_vs_t2=dict(audio=c5['models']['p2']['frozen_C5_summary'], t2lora=c5['models']['lora']['frozen_C5_summary']))
    MAIN = (('sector_K8_center', ('none', 'A', 'B')), ('noise_sigma10', ('none', 'A')))
    for tag, rb, ra, ck in (('cv_dev93', 'real_base', 'real_audio', 'checks_audio_vs_t2'), ('test_eval92_descriptive', 'real_base_te92', 'real_audio_te92', 'checks_audio_vs_t2_te92')):
        d = {}
        for cn, pols in MAIN:
            for pol in pols:
                rows = {}
                for route, src in (('P2', J[rb]['frontends']['spatialnet']['conditions'][cn]['routes']['P2']), ('WT2', J[rb]['frontends']['spatialnet']['conditions'][cn]['routes']['WT2']),
                                   ('AUDIO', J[ra]['frontends']['spatialnet']['conditions'][cn]['routes']['P2'])):
                    m = src['main'][pol]; rows[route] = {f: m[f] for f in m if f.endswith('_nat') or f.endswith('_eq')}
                pc = J[ck]['conditions'][cn][pol]['all']
                rows['T2LORA'] = {k: v for k, v in pc['models']['lora'].items() if k.endswith('_eq') or k.endswith('_nat')}
                rows['paired_audio_minus_t2lora'] = pc['paired']
                d[f'{cn}/{pol}'] = rows
        rep['mix'][f'c6_real_{tag}'] = d
    rep['risk_checks_descriptive'] = dict(c6_real_and_C5_audio_vs_p2=J['checks_p2']['gates'], C3=rep['mix']['C3']['devel_section6_rule_descriptive'],
                                          section11_by_actual_route={cn: J['checks_p2']['conditions'][cn]['none']['all']['models'] for cn in ('C5', 'sector_K8_center', 'noise_sigma10')})
    rep['other_comparisons']['C5_t2lora_minus_wt2'] = J['checks_t2']['conditions']['C5']['none']['all']['paired']
    rep['other_comparisons']['C5_audio_minus_p2'] = J['checks_p2']['conditions']['C5']['none']['all']['paired']
    out.mkdir(parents=True)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for c, p in rep['primary'].items():
        print('PRIMARY', c, f"audio {p['a_any']:.4f} t2lora {p['b_any']:.4f} delta {p['delta']:+.4f} 95% [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]"
              f" 98.33% [{p['ci9833'][0]:+.4f},{p['ci9833'][1]:+.4f}] significant={p['significant_bonferroni']}", flush=True)
    print('REPORT', out / 'report.json', sha(out / 'report.json'))


def _by_sid(R):
    by = collections.defaultdict(set)
    for r in R.values():
        by[r['slurp_id']].add(r['raw_output'])
    return by


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs'); ap.add_argument('--out'); ap.add_argument('--devel-check')
    a = ap.parse_args()
    if a.devel_check:
        devel_check(a.devel_check)
    else:
        main(a.inputs, a.out)
