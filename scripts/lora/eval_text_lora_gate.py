"""Step-0 text LoRA: checkpoint selection, gate and revision-1 reports (docs/lora_slu_plan_v2.md §4, §7.4, revision 1;
docs/lora_v2_runs.md §4). CPU only; reads raw outputs (format = C0-428 raw.jsonl) and training logs. Metric functions are
imported unchanged from scripts/qwen/eval_c0_devel428.py (summary/classify: slu-score-v2, single-recording "any gold value hit",
date recall, c1-error-taxonomy-v1) and binding-score-v1 / eval_c0_v1 (124 mix_v1 recordings, paired, by recording).

gate:   python -I -B scripts/lora/eval_text_lora_gate.py gate --baseline RAW428 --out DIR \
            --cand LR:STEP:SELECTION_RAW [...] --log LR:LOG_JSONL [...]
        Selection half only. Best = max any-gold-value-hit; ties -> smaller step, then smaller lr. Gate = paired bootstrap
        (best - baseline) over recordings, resampling the selection-half slurp_ids (sorted ascending) with replacement,
        B = 2000, np.random.default_rng(2026), rng.integers(0, n, size=(B, n)); statistic = recording-weighted mean difference
        of the resampled sentences; 95% interval = np.quantile(..., [0.025, 0.975]) (linear). Pass iff lower > 0.
        Exceptions (reported first, not counted as "no gain"): training loss not decreased (best checkpoint's group: mean loss of
        steps 408-444 >= mean of steps 1-37; last log entry per step), or parse failures (slu-score-v2 status != ok) > 2% of
        selection-half recordings.
report: python -I -B scripts/lora/eval_text_lora_gate.py report --baseline RAW428 --sel SELECTION_RAW --rep REPORT_RAW --out DIR
        Selection / report / all (428): any-gold-value-hit, date recall, taxonomy classes, intent acc, parse failures, 124 mix_v1
        recordings paired target-any by recording (C0 v1 = 0.782), for candidate and baseline, with paired bootstrap deltas
        (same procedure; report only).
Dry run: pass the baseline as the candidate -> every delta 0, every interval [0, 0].
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from binding_score_v1 import judge, vals  # noqa: E402
from eval_c0_devel428 import summary  # noqa: E402
from rescore_slu_v2 import score_item  # noqa: E402

B, SEED, PARSE_MAX = 2000, 2026, 0.02


def windows(total):
    """First / last 0.25 epoch (3 epochs): total 444 -> steps 1-37 / 408-444 (step 0); 1320 -> 1-110 / 1211-1320 (step 1)."""
    w = round(total / 3 * 0.25)
    return range(1, w + 1), range(total - w + 1, total + 1)
MIXCFG = PROJ / 'configs/mix_v1_devel100_v2.json'


def load(p):
    """One raw.jsonl, or several joined by commas (e.g. the 2 parts of a baseline)."""
    R = {}
    for l in (x for q in str(p).split(',') for x in open(q)):
        r = json.loads(l); assert r['file'] not in R, r['file']; R[r['file']] = r
    return R


def halves(R):
    ids = sorted({r['slurp_id'] for r in R.values()}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    return dict(selection={f for f, r in R.items() if r['slurp_id'] in sel}, report={f for f, r in R.items() if r['slurp_id'] not in sel})


def hits(R):
    return {f: int(judge(vals(r['ref_entities']), collections.Counter(), vals(score_item(r)['hyp']['entities']))['t_any']) for f, r in R.items()}


def boot(files, R, a, b):
    """Paired bootstrap of mean(a) - mean(b) over recordings in `files`, clustered by slurp_id."""
    by = collections.defaultdict(list)
    for f in files:
        by[R[f]['slurp_id']].append(a[f] - b[f])
    ids = sorted(by); s = np.array([sum(by[i]) for i in ids], float); c = np.array([len(by[i]) for i in ids], float)
    idx = np.random.default_rng(SEED).integers(0, len(ids), size=(B, len(ids)))
    st = s[idx].sum(1) / c[idx].sum(1); lo, hi = np.quantile(st, [0.025, 0.975])
    return dict(delta=float(s.sum() / c.sum()), ci95=[float(lo), float(hi)], lower_gt_0=bool(lo > 0), n_recordings=int(c.sum()), n_sentences=len(ids), B=B, seed=SEED)


def mix124(R):
    """binding-score-v1 paired target-any on the 124 mix_v1 recordings, averaged by recording (eval_c0_v1)."""
    L = json.load(open(MIXCFG))['groups']; per = collections.defaultdict(list)
    H = {f: vals(score_item(r)['hyp']['entities']) for f, r in R.items()}
    for g in L:
        VA, VB = vals(g['A']['entities']), vals(g['B']['entities'])
        for k, D_T, D_I in (('A', VA - VB, VB - VA), ('B', VB - VA, VA - VB)):
            f = g[k]['file']
            if f in H:
                per[f].append(judge(D_T, D_I, H[f]))
    return {f: sum(x['t_any'] for x in v) / len(v) for f, v in per.items()}


def metrics(R):
    s = summary(R)
    return dict(n=s['n'], any_gold_value_hit=s['any_gold_value_hit'], hit=s['hit'], gold_entities=s['gold_entities'], date_gold=s['date_gold'],
                date_recall=s['date_recall'], date_missed_classes=s['date_missed_classes'], classes=s['classes'],
                intent_strict_acc=s['slu_v2']['intent_strict_acc'], parse_failures=s['slu_v2']['parse_failures'],
                parse_failure_rate=s['slu_v2']['parse_failures'] / s['n'], zero_entity_outputs=s['slu_v2']['zero_entity_outputs'],
                command_refusals=s['slu_v2']['command_refusals'])


def check_same_inputs(C, BL):
    for f, r in C.items():
        b = BL[f]
        assert (r['slurp_id'], r['ref_sentence'], r['ref_intent'], r['ref_entities']) == (b['slurp_id'], b['ref_sentence'], b['ref_intent'], b['ref_entities']), f


def loss_check(log, total=444):
    FIRST, LAST = windows(total)
    last = {}
    for l in open(log):
        r = json.loads(l); last[r['step']] = r['loss']
    miss = [s for s in (*FIRST, *LAST) if s not in last]
    if miss:
        return dict(complete=False, missing_steps=miss[:10], n_missing=len(miss))
    f, la = float(np.mean([last[s] for s in FIRST])), float(np.mean([last[s] for s in LAST]))
    return dict(complete=True, first37_mean=f, last37_mean=la, not_decreased=la >= f)


def gate(a):
    BL = load(a.baseline); assert len(BL) == 428; H = halves(BL); sel = H['selection']; assert len(sel) == 214
    bh = hits(BL); cands = []
    for spec in a.cand:
        lr, st, p = spec.split(':', 2); C = load(p)
        if set(C) != sel:
            C = {f: C[f] for f in sel} if set(C) >= sel else None
        assert C is not None, f'{p} does not cover the selection half'
        check_same_inputs(C, BL); ch = hits(C)
        cands.append(dict(lr=lr, lr_value=float(lr), step=int(st), raw=p, any_gold_value_hit=sum(ch[f] for f in sel) / len(sel), _R=C, _h=ch))
    best = max(cands, key=lambda c: (c['any_gold_value_hit'], -c['step'], -c['lr_value']))
    C = best['_R']; m = metrics(C)
    logs = dict(x.split(':', 1) for x in a.log or [])
    lc = loss_check(logs[best['lr']], a.total_steps) if best['lr'] in logs else dict(complete=False, note='no log given')
    exc = dict(loss_not_decreased=lc.get('not_decreased'), loss_check=lc, parse_failure_rate=m['parse_failure_rate'],
               parse_failure_over_2pct=m['parse_failure_rate'] > PARSE_MAX)
    bs = boot(sel, BL, best['_h'], bh)
    if exc['loss_not_decreased'] or exc['parse_failure_over_2pct']:
        verdict = 'exception: report reason first (plan §7.4)'
    elif exc['loss_not_decreased'] is None:
        verdict = 'incomplete: loss log missing'
    else:
        verdict = 'pass' if bs['lower_gt_0'] else 'stop: LoRA 没有明显帮助'
    rep = dict(spec='docs/lora_slu_plan_v2.md §7.4 + revision 1 item 3; docs/lora_v2_runs.md §4', baseline=str(a.baseline),
               half='selection', n_recordings=len(sel), baseline_any_gold_value_hit=sum(bh[f] for f in sel) / len(sel),
               candidates=[{k: v for k, v in c.items() if not k.startswith('_')} for c in sorted(cands, key=lambda c: (c['lr_value'], c['step']))],
               best={k: v for k, v in best.items() if not k.startswith('_')}, best_metrics=m, exceptions=exc, bootstrap=bs, verdict=verdict)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'gate.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for c in rep['candidates']:
        print('CAND', c['lr'], c['step'], round(c['any_gold_value_hit'], 4))
    print('BEST', json.dumps(rep['best'])); print('BOOT', json.dumps(bs)); print('EXC', json.dumps(exc)); print('VERDICT', verdict)


def report(a):
    BL = load(a.baseline); assert len(BL) == 428; H = halves(BL)
    S, P = load(a.sel), load(a.rep)
    S = {f: S[f] for f in H['selection']}; P = {f: P[f] for f in H['report']}
    C = {**S, **P}; assert set(C) == set(BL); check_same_inputs(C, BL)
    ch, bh = hits(C), hits(BL); cm, bm = mix124(C), mix124(BL); assert len(bm) == 124
    rep = dict(spec='docs/lora_slu_plan_v2.md revision 1 item 4', baseline=str(a.baseline), selection_raw=str(a.sel), report_raw=str(a.rep), parts={})
    for k, fs in (('selection', H['selection']), ('report', H['report']), ('all', set(BL))):
        cs, bsub = {f: C[f] for f in fs}, {f: BL[f] for f in fs}; mf = [f for f in fs if f in bm]
        x = dict(candidate=metrics(cs), baseline=metrics(bsub))
        x['any_gold_value_hit_bootstrap'] = boot(fs, BL, ch, bh)
        x['mix124_paired'] = dict(n=len(mf), candidate=sum(cm[f] for f in mf) / len(mf), baseline=sum(bm[f] for f in mf) / len(mf),
                                  bootstrap=boot(mf, BL, cm, bm))
        x['date_recall_delta'] = x['candidate']['date_recall'] - x['baseline']['date_recall']
        x['classes_delta'] = {c: x['candidate']['classes'][c] - x['baseline']['classes'][c] for c in x['candidate']['classes']}
        rep['parts'][k] = x
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for k, x in rep['parts'].items():
        c, b = x['candidate'], x['baseline']
        print(k, 'any', round(c['any_gold_value_hit'], 4), 'base', round(b['any_gold_value_hit'], 4), 'boot', json.dumps(x['any_gold_value_hit_bootstrap']))
        print(' mix124', json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x['mix124_paired'].items() if kk != 'bootstrap'}),
              'delta', round(x['mix124_paired']['bootstrap']['delta'], 4), x['mix124_paired']['bootstrap']['ci95'])
        print(' date', round(c['date_recall'], 4), 'base', round(b['date_recall'], 4), 'classes', c['classes'], 'delta', x['classes_delta'])
        print(' intent', round(c['intent_strict_acc'], 4), 'parse_fail', c['parse_failures'])


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    g = sp.add_parser('gate'); g.add_argument('--baseline', required=True); g.add_argument('--cand', nargs='+', required=True)
    g.add_argument('--log', nargs='*'); g.add_argument('--total-steps', type=int, default=444); g.add_argument('--out', required=True)
    r = sp.add_parser('report'); r.add_argument('--baseline', required=True); r.add_argument('--sel', required=True)
    r.add_argument('--rep', required=True); r.add_argument('--out', required=True)
    a = ap.parse_args(); gate(a) if a.cmd == 'gate' else report(a)


if __name__ == '__main__':
    main()
