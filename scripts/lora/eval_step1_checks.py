"""Step-1 checks that do not depend on SpatialNet (docs/lora_v2_runs.md §20-21; plan §6, §11). CPU, reads raw outputs only.
Audio LoRA best checkpoint (1e-4 step 440) vs the existing P2 baselines.
  C4-direct / C3 (mix_v1, 200 samples): per-sample binding-score-v1 judge exactly as eval_c3_v1 / eval_c4_v1
  (D_T = V_target - V_interferer, D_I = V_interferer - V_target; hypothesis values from slu-score-v2 score_item).
  Decisions on ALL 200 (§20-3), point estimates: C4-direct target-any >= P2 - 0.03; C3 only-interferer <= P2 + 0.02 and
  target-any >= P2 - 0.03. Selection / report halves = c6-delta-v1 group split (sorted group ids, random.Random(2026).shuffle,
  first 50 = selection), reported only. If only-interferer rises at all: §11 entity counts (predicted entities, D_I hits, D_T hits;
  C3 has no route selection -> all samples).
  C1 report half and C1-R (selection / report / all): eval_c0_devel428.summary + paired bootstrap (report only).
Usage: python -I -B scripts/lora/eval_step1_checks.py OUT_DIR
"""
import collections
import glob
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/slurp')); sys.path.insert(0, str(PROJ / 'scripts/lora'))
from binding_score_v1 import judge, vals  # noqa: E402
from eval_c0_devel428 import summary  # noqa: E402
from eval_text_lora_gate import boot, hits  # noqa: E402
from rescore_slu_v2 import score_item  # noqa: E402

O = PROJ / 'outputs'
LIST = PROJ / 'configs/mix_v1_devel100_v2.json'
BASE = dict(c4direct=O / 'qwen_omni_c4_direct/22015148/C4_direct/raw.jsonl', c3=O / 'qwen_omni_c3_p2/22014089/C3/raw.jsonl')
P2_CLEAN = 'lora_base_audio/p2_devel_clean/*/P2/raw.jsonl'; P2_REV = 'lora_base_audio/p2_devel_rev/*/P2/raw.jsonl'


def one(pattern):
    ps = sorted(glob.glob(str(O / pattern))); assert len(ps) == 1, (pattern, ps)
    return ps[0]


def load(paths):
    R = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l); assert r['file'] not in R; R[r['file']] = r
    return R


def mix_rows(R, L):
    gl = {g['group']: g for g in L}; out = {}
    for f, r in R.items():
        g = gl[r['group']]; t = r['target']; i = 'B' if t == 'A' else 'A'
        VT, VI = vals(g[t]['entities']), vals(g[i]['entities']); s = score_item(r); H = vals(s['hyp']['entities'])
        j = judge(VT - VI, VI - VT, H)
        out[f] = dict(group=r['group'], cat=j['cat'], t_any=j['t_any'], t_all=j['t_all'], parse_fail=s['status'] != 'ok',
                      n_pred=sum(H.values()), di_hits=sum(((VI - VT) & H).values()), dt_hits=sum(((VT - VI) & H).values()))
    return out


def msum(rows):
    n = len(rows)
    return dict(n=n, target_any=sum(r['t_any'] for r in rows) / n, target_all=sum(r['t_all'] for r in rows) / n,
                **{c: sum(r['cat'] == c for r in rows) / n for c in ('only_target', 'only_interferer', 'both', 'neither')},
                parse_failures=sum(r['parse_fail'] for r in rows))


def ents(rows):
    n = len(rows)
    return dict(n=n, pred_entities=sum(r['n_pred'] for r in rows), pred_per_sample=sum(r['n_pred'] for r in rows) / n,
                interferer_value_hits=sum(r['di_hits'] for r in rows), target_value_hits=sum(r['dt_hits'] for r in rows))


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    L = json.load(open(LIST))['groups']
    gids = sorted(g['group'] for g in L); random.Random(2026).shuffle(gids); selg = set(gids[:50])
    rep = dict(spec='docs/lora_v2_runs.md §20-21; plan §6, §11', checkpoint='step1/lr1e-4/ckpt_step440', mix={}, single={})
    for cond in ('c4direct', 'c3'):
        lp = one(f'lora_audio_mix/lr1e-4/*/step440/{cond}/raw.jsonl'); Lr, Br = load([lp]), load([BASE[cond]])
        assert set(Lr) == set(Br) and len(Lr) == 200
        assert all(Lr[f]['prompt_text'] == Br[f]['prompt_text'] for f in Lr), 'prompt mismatch'
        lr_, br_ = mix_rows(Lr, L), mix_rows(Br, L); x = dict(lora_raw=lp, baseline_raw=str(BASE[cond]))
        for h, keep in (('all', lambda g: True), ('selection', lambda g: g in selg), ('report', lambda g: g not in selg)):
            fs = [f for f in Lr if keep(lr_[f]['group'])]
            x[h] = dict(lora=msum([lr_[f] for f in fs]), p2=msum([br_[f] for f in fs]))
        a, b = x['all']['lora'], x['all']['p2']
        if cond == 'c4direct':
            x['decision'] = dict(rule='target_any >= P2 - 0.03 (all 200)', delta_target_any=a['target_any'] - b['target_any'],
                                 passes=a['target_any'] >= b['target_any'] - 0.03 - 1e-12)
        else:
            x['decision'] = dict(rule='only_interferer <= P2 + 0.02 and target_any >= P2 - 0.03 (all 200)',
                                 delta_only_interferer=a['only_interferer'] - b['only_interferer'], delta_target_any=a['target_any'] - b['target_any'],
                                 passes=(a['only_interferer'] <= b['only_interferer'] + 0.02 + 1e-12) and (a['target_any'] >= b['target_any'] - 0.03 - 1e-12))
        x['only_interferer_rose'] = a['only_interferer'] > b['only_interferer']
        x['entity_counts_all_samples'] = dict(lora=ents(list(lr_.values())), p2=ents(list(br_.values())))  # §11 (reported always; required if rose)
        rep['mix'][cond] = x
    # single-talker: C1 report half, C1-R
    c1l = load([one('lora_audio_eval/lr1e-4/*/step440/devel_clean_report/raw.jsonl')])
    c1sel = load([one('lora_audio_eval/lr1e-4/22025280/step440/devel_clean_selection/raw.jsonl')])
    rvl = load([one('lora_audio_eval/lr1e-4/*/step440/devel_rev_selection/raw.jsonl'), one('lora_audio_eval/lr1e-4/*/step440/devel_rev_report/raw.jsonl')])
    P2c = load(sorted(glob.glob(str(O / P2_CLEAN)))); P2r = load(sorted(glob.glob(str(O / P2_REV))))
    assert len(P2c) == len(P2r) == 428 and len(rvl) == 428 and len(c1l) == 214 and len(c1sel) == 214
    ids = sorted({r['slurp_id'] for r in P2c.values()}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    H = dict(selection={f for f in P2c if P2c[f]['slurp_id'] in sel}, report={f for f in P2c if P2c[f]['slurp_id'] not in sel}, all=set(P2c))
    assert set(c1l) == H['report'] and set(c1sel) == H['selection']
    keys = ('any_gold_value_hit', 'hit', 'gold_entities', 'date_recall', 'classes')
    def part(Lx, Bx, fs):
        s, b = summary({f: Lx[f] for f in fs}), summary({f: Bx[f] for f in fs})
        return dict(lora={k: s[k] for k in keys} | dict(intent=s['slu_v2']['intent_strict_acc'], parse_failures=s['slu_v2']['parse_failures']),
                    p2={k: b[k] for k in keys} | dict(intent=b['slu_v2']['intent_strict_acc'], parse_failures=b['slu_v2']['parse_failures']),
                    bootstrap=boot(fs, Bx, hits({f: Lx[f] for f in fs}), hits({f: Bx[f] for f in fs})))
    C1all = {**c1sel, **c1l}
    rep['single']['C1'] = {h: part(C1all, P2c, fs) for h, fs in H.items()}
    rep['single']['C1R'] = {h: part(rvl, P2r, fs) for h, fs in H.items()}
    (out / 'eval_step1_checks.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for cond, x in rep['mix'].items():
        for h in ('all', 'selection', 'report'):
            a, b = x[h]['lora'], x[h]['p2']
            print(cond, h, 'any %.3f vs %.3f | only_int %.3f vs %.3f | only_tgt %.3f vs %.3f | pf %d vs %d' % (a['target_any'], b['target_any'], a['only_interferer'], b['only_interferer'],
                                                                                                         a['only_target'], b['only_target'], a['parse_failures'], b['parse_failures']))
        print(' DECISION', json.dumps(x['decision']), 'oi_rose', x['only_interferer_rose'], 'ENT', json.dumps(x['entity_counts_all_samples']))
    for c, v in rep['single'].items():
        for h, x in v.items():
            print(c, h, 'any %.3f vs %.3f' % (x['lora']['any_gold_value_hit'], x['p2']['any_gold_value_hit']), 'delta %.3f' % x['bootstrap']['delta'], [round(q, 3) for q in x['bootstrap']['ci95']],
                  'date %.3f vs %.3f' % (x['lora']['date_recall'], x['p2']['date_recall']), 'intent %.3f vs %.3f' % (x['lora']['intent'], x['p2']['intent']), 'pf', x['lora']['parse_failures'])


if __name__ == '__main__':
    main(sys.argv[1])
