"""FSC pilot v2 scoring (docs/fsc_pilot_plan.md "冻结语义评分"; configs/fsc_semantic_mapping.json frozen 2026-11-09). CPU only.

Applies the frozen rules without adding any mapping:
  * FSC target = {action, object, location}; 'none' is not a value. Values are compared after normalize_v2 (the SLURP scorer
    normalisation) with exact full-value equality, multiset (binding_score_v1.vals), types ignored ("concept/value" layer),
    because configs/fsc_semantic_mapping.json contains no audited FSC-slot -> SLURP-type or SLURP-intent -> FSC-triple table.
  * intent_exact and intent_family need such a table -> NOT computable; reported as coverage 0, never approximated.
  * value-level slot metrics: per-slot recall (action / object / location), macro slot recall, slot-value precision / recall.
  * target_any: >= 1 target slot value equals a predicted entity value exactly.
  * spatial / direction_uncertainty: binding_score_v1.judge on D_T = T - I, D_I = I - T (I = the pair partner's target);
    pairs with an empty D_T or D_I are excluded from binding metrics (count reported). wrong_speaker_only = only_interferer.
  * parse failure (slu_normalize.parse status != ok), refusal (postproc_c1_ext.REFUSAL on unparsed output),
    direction_uncertainty answered = route manifest within_tau (frozen C6 tau = 16 deg); coverage and answered-only risk.
  * Coverage of outputs by the frozen value layer: share of predicted entity values that equal any FSC slot value.
  * Bootstrap (descriptive): B=2000, np.random.default_rng(2026), items resampled by pair (spatial) or utterance (single).
Usage: python -I -B scripts/fsc/score_pilot.py OUT_DIR base=PRED_DIR audio_lora=PRED_DIR t2_lora=PRED_DIR
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for d in ('scripts/qwen', 'scripts/slurp'):
    sys.path.insert(0, str(ROOT / d))
from binding_score_v1 import judge, vals  # noqa: E402
from postproc_c1_ext import REFUSAL  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import parse, wer_counts  # noqa: E402

O = ROOT / 'outputs/fsc_pilot'
TASKS_SHA = '9f9cc65db6ca3b81f1f208a8ee4b9a1d258fcbd265fa9ed923d35af850fd9e36'
MAP = ROOT / 'configs/fsc_semantic_mapping.json'
SLOTS = ('action', 'object', 'location')
CONDS = ('clean', 'reverb', 'spatial', 'direction_uncertainty')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gold(t):
    return {s: normalize_v2(t[s]) for s in SLOTS if t[s] != 'none'}


def boot(x, groups):
    x = np.asarray(x, float); g = np.asarray(groups); u = sorted(set(groups)); ix = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(2026); z = []
    for _ in range(2000):
        pick = np.concatenate([ix[k] for k in rng.choice(u, size=len(u), replace=True)]); z.append(x[pick].mean())
    return dict(mean=float(x.mean()), ci95=[float(v) for v in np.quantile(z, [.025, .975])], n=len(x), groups=len(u))


def main(out, *specs):
    out = Path(out); assert not out.exists()
    assert sha(O / 'pilot_tasks.jsonl') == TASKS_SHA
    T = [json.loads(x) for x in open(O / 'pilot_tasks.jsonl')]; assert len(T) == 256
    R = {r['sample_id']: r for r in map(json.loads, open(O / 'route_manifest/route_manifest.jsonl'))}
    U = {(u['sample_id'], u['query_delta_deg']): u for u in map(json.loads, open(O / 'route_manifest/direction_uncertainty_manifest.jsonl'))}
    by_pair = collections.defaultdict(dict)
    for t in T:
        if t['condition'] == 'spatial':
            by_pair[R[t['sample_id']]['pair_id']][R[t['sample_id']]['swap']] = t
    vocab = {normalize_v2(t['target'][s]) for t in T for s in SLOTS if t['target'][s] != 'none'}
    routes = dict(s.split('=') for s in specs); assert set(routes) == {'base', 'audio_lora', 't2_lora'}
    rep = dict(spec='docs/fsc_pilot_plan.md 冻结语义评分', mapping_file=str(MAP.relative_to(ROOT)), mapping_sha256=sha(MAP), tasks_sha256=TASKS_SHA,
               not_computable=dict(intent_exact='requires an audited SLURP-output -> FSC (action, object, location) mapping; none frozen; coverage 0/256',
                                   intent_family='auxiliary; mapping "requires manual audited mapping" (not frozen); not computed'),
               routes={}, inputs={})
    per = {}
    for rt, d in routes.items():
        d = Path(d); P = [json.loads(x) for x in open(d / 'predictions.jsonl')]; rep['inputs'][rt] = {str(d / 'predictions.jsonl'): sha(d / 'predictions.jsonl')}
        assert len(P) == 256 and all(p['index'] == i and p['condition'] == t['condition'] and p['pilot_index'] == t['pilot_index'] and p['audio_sha256'] == t['audio_sha256']
                                     for i, (p, t) in enumerate(zip(P, T)))
        rows = []
        for p, t in zip(P, T):
            obj, st = parse(p['raw_output']); H = vals(obj['entities']) if obj else collections.Counter()
            G = gold(t['target']); row = dict(condition=t['condition'], pilot_index=t['pilot_index'], sample_id=t.get('sample_id'), delta=t.get('query_delta_deg'),
                                              parse_fail=st != 'ok', refusal=st == 'no_json' and bool(REFUSAL.search(p['raw_output'])),
                                              n_pred=sum(H.values()), n_pred_in_vocab=sum(c for v, c in H.items() if v in vocab),
                                              slot_hit={s: (G[s] in H) for s in G}, target_any=any(G[s] in H for s in G), n_gold=len(G),
                                              n_gold_hit=sum(G[s] in H for s in G), n_pred_match_gold=sum(c for v, c in H.items() if v in set(G.values())))
            tr = obj['transcript'] if obj else ''; e, n = wer_counts(normalize_v2(t['transcript']), normalize_v2(tr)); row.update(wer_e=e, wer_n=n)
            if rt == 't2_lora':
                e, n = wer_counts(normalize_v2(t['transcript']), normalize_v2(p['asr'] or '')); row.update(asr_wer_e=e, asr_wer_n=n)
            if t['condition'] in ('spatial', 'direction_uncertainty'):
                r = R[t['sample_id']]; other = by_pair[r['pair_id']][2 if r['swap'] == 1 else 1]
                VT, VI = vals([(s, v) for s, v in t['target'].items() if v != 'none']), vals([(s, v) for s, v in other['target'].items() if v != 'none'])
                DT, DI = VT - VI, VI - VT; row['binding_ok'] = bool(DT) and bool(DI)
                if row['binding_ok']:
                    row['cat'] = judge(DT, DI, H)['cat']
                row['pair_id'] = r['pair_id']
                if t['condition'] == 'direction_uncertainty':
                    row['answered'] = bool(U[(t['sample_id'], t['query_delta_deg'])]['within_tau'])
            rows.append(row)
        per[rt] = rows
        res = {}
        for c in CONDS:
            X = [r for r in rows if r['condition'] == c]; grp = [r.get('pair_id') or r['pilot_index'] for r in X]
            g = dict(n=len(X), target_any=boot([r['target_any'] for r in X], grp),
                     slot_recall={s: (sum(r['slot_hit'][s] for r in X if s in r['slot_hit']) / max(1, sum(s in r['slot_hit'] for r in X)), sum(s in r['slot_hit'] for r in X)) for s in SLOTS},
                     value_recall=sum(r['n_gold_hit'] for r in X) / sum(r['n_gold'] for r in X),
                     value_precision=(sum(r['n_pred_match_gold'] for r in X) / sum(r['n_pred'] for r in X)) if sum(r['n_pred'] for r in X) else None,
                     parse_fail=sum(r['parse_fail'] for r in X), refusal=sum(r['refusal'] for r in X),
                     zero_entity=sum(r['n_pred'] == 0 for r in X), pred_values=sum(r['n_pred'] for r in X),
                     output_coverage_by_value_layer=(sum(r['n_pred_in_vocab'] for r in X) / sum(r['n_pred'] for r in X)) if sum(r['n_pred'] for r in X) else None,
                     transcript_wer=sum(r['wer_e'] for r in X) / sum(r['wer_n'] for r in X))
            g['macro_slot_recall'] = float(np.mean([v[0] for v in g['slot_recall'].values() if v[1]]))
            if rt == 't2_lora':
                g['whisper_wer'] = sum(r['asr_wer_e'] for r in X) / sum(r['asr_wer_n'] for r in X)
            if c in ('spatial', 'direction_uncertainty'):
                B = [r for r in X if r['binding_ok']]
                g['binding'] = dict(n_distinguishable=len(B), excluded_no_distinct_values=len(X) - len(B),
                                    **{k: sum(r['cat'] == k for r in B) / len(B) for k in ('only_target', 'only_interferer', 'both', 'neither')})
                g['wrong_speaker_only'] = g['binding']['only_interferer']
            if c == 'direction_uncertainty':
                A = [r for r in X if r['answered']]
                g['by_delta'] = {str(d): dict(n=sum(r['delta'] == d for r in X), target_any=float(np.mean([r['target_any'] for r in X if r['delta'] == d])),
                                              coverage=float(np.mean([r['answered'] for r in X if r['delta'] == d]))) for d in sorted({r['delta'] for r in X})}
                g['abstention'] = dict(rule='route manifest within_tau (frozen C6 tau=16)', coverage=len(A) / len(X), answered=len(A),
                                       answered_target_any=float(np.mean([r['target_any'] for r in A])) if A else None,
                                       answered_wrong_speaker_only=(sum(r.get('cat') == 'only_interferer' for r in A if r['binding_ok']) / max(1, sum(r['binding_ok'] for r in A))) if A else None)
            res[c] = g
        rep['routes'][rt] = res
    # descriptive paired difference, same direction as the SLURP main comparison (Audio LoRA - Whisper+T2-LoRA)
    rep['paired_audio_minus_t2lora_target_any'] = {}
    for c in CONDS:
        A = [r for r in per['audio_lora'] if r['condition'] == c]; B = [r for r in per['t2_lora'] if r['condition'] == c]
        rep['paired_audio_minus_t2lora_target_any'][c] = boot([a['target_any'] - b['target_any'] for a, b in zip(A, B)], [a.get('pair_id') or a['pilot_index'] for a in A])
    out.mkdir(parents=True)
    (out / 'fsc_pilot_scores.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    with open(out / 'per_item.jsonl', 'w') as fh:
        for rt, rows in per.items():
            for r in rows:
                fh.write(json.dumps(dict(route=rt, **r)) + '\n')
    for rt, res in rep['routes'].items():
        for c, g in res.items():
            print(rt, c, 'target_any %.3f [%.3f,%.3f]' % (g['target_any']['mean'], *g['target_any']['ci95']), 'slot', {s: round(v[0], 3) for s, v in g['slot_recall'].items()},
                  'pf', g['parse_fail'], 'cov %.3f' % g['output_coverage_by_value_layer'], 'bind', {k: round(v, 3) for k, v in g.get('binding', {}).items()})
    print('PAIRED', json.dumps({c: [round(v['mean'], 4), [round(x, 4) for x in v['ci95']]] for c, v in rep['paired_audio_minus_t2lora_target_any'].items()}))
    print('SHA256', sha(out / 'fsc_pilot_scores.json'))


if __name__ == '__main__':
    main(*sys.argv[1:])
