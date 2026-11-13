"""Frozen §6 final SpatialNet gates, paired C5/C6 intervals, and §11 diagnostics.
Uses original score/summ, sector/choose/wrap and frozen seeds/weights. The entity
diagnosis groups ACTUAL selected-reference agreement, independently of the sector
indistinguishability penalty. Counts for noise sum over all 10 query repetitions.
--self-check requires identical baseline/candidate paths and verifies zero deltas.
SLURP test (2026-11-12, docs/slurp_test_freeze_v1.md): env SLURP_EVAL_SPLIT=test -> test list/manifest, n = 2 x groups,
half "all" only, no C6delta queries; paired results additionally carry ci9833 (0.8333%/99.1667% quantiles of the same draws,
Bonferroni over 3 primary comparisons). Devel default unchanged.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen'))
import eval_c6_real as real
from eval_c4_v1 import score, summ, parse, vals
import split_cfg
TEST = split_cfg.SPLIT == 'test'


def load(path, key='file'):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    N = sum(1 for _ in open(real.MANIFEST)) if TEST else 200
    assert len(rows) == N and len({r[key] for r in rows}) == N
    return {r[key]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    for name in ['baseline-selected', 'baseline-other', 'lora-selected', 'lora-other',
                 'selection-dir', 'baseline-real', 'lora-real', 'out']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--route-kind', choices=['p2', 't2', 'audio_vs_t2'], default='p2')
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--natural', choices=('cv_dev93', 'test_eval92'), default='cv_dev93')  # test_eval92: descriptive only (freeze §0-5)
    ap.add_argument('--ci9833', action='store_true')  # add Bonferroni 98.33% intervals (always on for the test split)
    a = ap.parse_args()
    if a.natural != 'cv_dev93':
        assert TEST; print('NATURAL', split_cfg.set_natural(a.natural), flush=True)
    assert not a.out.exists()
    paths = {'p2': [a.baseline_selected, a.baseline_other], 'lora': [a.lora_selected, a.lora_other]}
    raw_name = 'P2' if a.route_kind == 'p2' else 'T2'
    raw = {'p2': [load(p / f"{'P2' if a.route_kind == 'audio_vs_t2' else raw_name}/raw.jsonl") for p in paths['p2']],
           'lora': [load(p / f'T2/raw.jsonl') for p in paths['lora']]} if a.route_kind == 'audio_vs_t2' else {m: [load(p / f'{raw_name}/raw.jsonl') for p in ps] for m, ps in paths.items()}
    if a.self_check:
        assert paths['p2'] == paths['lora'] and a.baseline_real == a.lora_real
    else:
        for i in range(2):
            bc = json.loads((paths['p2'][i] / 'config.json').read_text())
            lc = json.loads((paths['lora'][i] / 'config.json').read_text())
            if a.route_kind in ('p2','audio_vs_t2'):
                if a.route_kind == 'p2':
                    assert bc['audio_sha256'] == lc['audio_sha256'] and len(bc['audio_sha256']) == (len(raw['p2'][i]) if TEST else 200)
                    assert bc['prompt_P2'] == lc['prompt_P2']
                else:
                    assert len(bc['audio_sha256']) == (len(raw['p2'][i]) if TEST else 200)
                if a.route_kind == 'p2': assert lc['adapter'].endswith('/step1/lr1e-4/ckpt_step440')
                else:
                    assert lc['run_dir'].endswith('/step2/lr1e-4') and lc['step'] == 660
            else:
                assert bc['prompt_T2'] == lc['prompt_T2']
                assert lc['run_dir'].endswith('/step2/lr1e-4') and lc['step'] == 660
            for k, br in raw['p2'][i].items():
                lr = raw['lora'][i][k]
                fields = ['file', 'group', 'swap', 'target', 'ref_sentence', 'ref_intent', 'ref_entities']
                if a.route_kind == 'p2': fields += ['prompt_text', 'stream']
                for field in fields:
                    assert lr[field] == br[field], (k, field)
                if a.route_kind == 't2':
                    assert lr['asr_text'] == br['asr_text'], (k, 'asr_text')
    L = {g['group']: g for g in json.load(open(split_cfg.LIST))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(real.MANIFEST))}
    keys = sorted(man); n = len(keys); NG = len(L); assert n == 2 * NG and (TEST or n == 200)
    sel = load(a.selection_dir / 'selection.jsonl', 'key')
    ref = load(a.selection_dir / 'per_sample.jsonl', 'key')
    assert set(sel) == set(ref) == set(keys)
    gid = np.array([man[k]['group'] for k in keys])
    th1 = np.array([L[g]['rir']['theta1'] for g in gid]); th2 = np.array([L[g]['rir']['theta2'] for g in gid])
    az = np.array([sel[k]['doa'] for k in keys]); selected = np.array([sel[k]['stream'] for k in keys])
    refs = np.array([ref[k]['ref_stream'] for k in keys])
    strata = [L[g]['rir']['stratum'] for g in gid]
    weights = np.array([real.NATURAL[s] / sum(real.NATURAL.values()) / strata.count(s) for s in strata])
    group_ids = sorted(L); random.Random(2026).shuffle(group_ids)
    halves = {'all': np.ones(n, bool), 'selection': np.isin(gid, group_ids[:50]), 'report': np.isin(gid, group_ids[50:])}
    if TEST:
        halves = {'all': halves['all']}
    index_by_group = {g: np.flatnonzero(gid == g) for g in sorted(L)}
    boot = {}
    for h, mask in halves.items():
        by = [sorted(set(gid[mask & np.array([s == st for s in strata])])) for st in real.STRATA]
        rng = np.random.default_rng(2026)
        boot[h] = np.array([np.concatenate([index_by_group[g] for gs in by for g in rng.choice(gs, size=len(gs), replace=True)]) for _ in range(2000)])
    scored = {m: [{r['key']: r for r in score(rs, L, man)} for rs in streams] for m, streams in raw.items()}
    data = {}
    for m in raw:
        data[m] = []
        for st in range(2):
            result = {}
            for k in keys:
                g = L[man[k]['group']]; t = man[k]['target']; i = 'B' if t == 'A' else 'A'
                obj, status = parse(raw[m][st][k]['raw_output'])
                H = vals(obj['entities']) if obj else vals([])
                T, I = vals(g[t]['entities']), vals(g[i]['entities'])
                result[k] = dict(scored[m][st][k], pred=sum(H.values()),
                                 dt=sum(((T-I)&H).values()), di=sum(((I-T)&H).values()))
            data[m].append(result)
    queries = {'C5': (th1[None], np.zeros((1,n),bool)),
               'sector_K8_center': (np.array([[real.sector(t,8,'center')[1] for t in th1]]),
                                    np.array([[real.sector(t,8,'center')[0] == real.sector(i,8,'center')[0] for t,i in zip(th1,th2)]]))}
    for d in (() if TEST else (10, -10, 20, -20)):
        queries[f'C6delta_{d:+d}'] = (np.array([[real.wrap(t + d) for t in th1]]), np.zeros((1,n), bool))
    noise_queries = []
    for r in range(10):
        eps = np.random.default_rng(np.random.SeedSequence([2026,10,r])).normal(0,10,NG)
        noise_queries.append([real.wrap(t + eps[g]) for t,g in zip(th1,gid)])
    queries['noise_sigma10'] = (np.array(noise_queries), np.zeros((10,n),bool))
    stored = {m: json.loads(p.read_text())['frontends']['spatialnet']['conditions'] for m,p in [('p2',a.baseline_real),('lora',a.lora_real)]}
    report = {'spec':'lora_slu_plan_v2 §4/§6; lora_v2_runs §11/§20; c6-real-v1', 'route_kind': a.route_kind, 'B':2000, 'seed':2026,
              'self_check':a.self_check, 'inputs':{m:[str(p) for p in ps] for m,ps in paths.items()}, 'conditions':{}}
    def mean_rate(x, mask, w):
        den=(mask*w).sum(1)
        return float(np.mean((x*mask*w).sum(1)/den)) if np.all(den>0) else None
    def interval(x, mask, w, draws):
        den=(mask[:,draws]*w[draws]).sum(2)
        num=(x[:,draws]*mask[:,draws]*w[draws]).sum(2)
        samples=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
        samples=np.nanmean(samples,axis=0)
        return np.nanquantile(samples,[.025,.975]).tolist()
    def interval9833(x, mask, w, draws):  # test only: same draws and formula, Bonferroni (3 primary comparisons) quantiles
        den=(mask[:,draws]*w[draws]).sum(2)
        num=(x[:,draws]*mask[:,draws]*w[draws]).sum(2)
        samples=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
        samples=np.nanmean(samples,axis=0)
        return np.nanquantile(samples,[.05/3/2,1-.05/3/2]).tolist()
    skipped=[]
    for cn,(Q,ind) in queries.items():
        # 2026-11-12 fix (S0 item 4): the p2 route asserts every non-C5 query against the stored c6-real conditions; C6delta queries
        # (added in 4580ad6) have no stored c6-real condition, so they are skipped for p2 and listed under 'skipped_queries'.
        if a.route_kind == 'p2' and cn != 'C5' and not all(cn in stored[m] for m in stored):
            skipped.append(cn); continue
        chosen=np.array([[real.choose(az[i],q) for i,q in enumerate(qr)] for qr in Q])
        if cn == 'C5':
            assert np.array_equal(chosen[0], selected)
        margin=np.array([[abs(real.wrap(q-az[i][1-chosen[r,i]]))-abs(real.wrap(q-az[i][chosen[r,i]])) for i,q in enumerate(qr)] for r,qr in enumerate(Q)])
        wrong=chosen!=refs[None]; policies={'none':np.ones_like(chosen,bool)}
        if cn!='C5': policies['A']=margin>=(16 if cn.startswith('C6delta_') else real.TAU_A)
        if cn.startswith('sector'):
            policies['B']=np.array([[not (real.sector(az[i][0],8,'center')[0]==real.sector(q,8,'center')[0] and real.sector(az[i][1],8,'center')[0]==real.sector(q,8,'center')[0]) for i,q in enumerate(Q[0])]])
        arrays={}
        for m in data:
            rows=[[data[m][0 if chosen[r,i]==selected[i] else 1][k] for i,k in enumerate(keys)] for r in range(len(Q))]
            arrays[m]={f:np.array([[row[f] for row in rr] for rr in rows]) for f in ['t_any','t_all','cat','pred','dt','di']}
            arrays[m]['target_any']=arrays[m]['t_any'] & ~ind
            arrays[m]['only_interferer']=arrays[m]['cat']=='only_interferer'
        result={}
        for policy,answered in policies.items():
            result[policy]={}
            for h,hm in halves.items():
                mask=answered & hm[None]
                stats={'answered_observations':int(mask.sum()),'repetitions':len(Q),'models':{},'paired':{}}
                for label,w in [('eq',np.ones(n)),('nat',weights)]:
                    stats['coverage_'+label]=mean_rate(answered,np.broadcast_to(hm,answered.shape),w)
                    stats['selection_error_main_'+label]=mean_rate(wrong|ind,mask,w)
                    stats['selection_error_actual_'+label]=mean_rate(wrong,mask,w)
                for m,arr in arrays.items():
                    ms={f+'_'+label:mean_rate(arr[f],mask,w) for f in ['target_any','only_interferer'] for label,w in [('eq',np.ones(n)),('nat',weights)]}
                    ms['by_actual_route']={}
                    for label,rm in [('correct',~wrong),('wrong',wrong)]:
                        pick=mask & rm; count=int(pick.sum())
                        ms['by_actual_route'][label]={'n_observations':count,'pred_entities':int(arr['pred'][pick].sum()),
                            'pred_per_observation':float(arr['pred'][pick].mean()) if count else None,
                            'target_value_hits':int(arr['dt'][pick].sum()),'interferer_value_hits':int(arr['di'][pick].sum()),
                            'binding_counts':{c:int(((arr['cat']==c)&pick).sum()) for c in ['only_target','only_interferer','both','neither']}}
                    if cn=='C5':
                        rs=[scored[m][0][k] for i,k in enumerate(keys) if hm[i]]
                        ms['frozen_C5_summary']=summ(rs)
                        ms['other_stream_report_only']=summ([scored[m][1][k] for i,k in enumerate(keys) if hm[i]])
                    if cn!='C5' and h=='all' and a.route_kind == 'p2':
                        frozen=stored[m][cn]['routes']['P2']['main'][policy]
                        for f in ['target_any','only_interferer']:
                            for label in ['eq','nat']: assert abs(ms[f+'_'+label]-frozen[f+'_'+label]['mean'])<1e-12,(m,cn,policy,f,label)
                        for label in ['eq','nat']:
                            assert abs(stats['coverage_'+label]-frozen['coverage_'+label]['mean'])<1e-12
                            assert abs(stats['selection_error_main_'+label]-frozen['sel_error_'+label]['mean'])<1e-12
                    stats['models'][m]=ms
                for f in ['target_any','only_interferer']:
                    # Historical p2/t2 reports used candidate-minus-baseline (T2-Audio).
                    # The formal comparison table defines Audio-T2, so expose that order explicitly.
                    diff=(arrays['p2'][f].astype(float)-arrays['lora'][f].astype(float)
                          if a.route_kind == 'audio_vs_t2' else arrays['lora'][f].astype(float)-arrays['p2'][f].astype(float))
                    stats['paired'][f]={label:dict(delta=mean_rate(diff,mask,w),ci95=interval(diff,mask,w,boot[h]),**(dict(ci9833=interval9833(diff,mask,w,boot[h])) if (TEST or a.ci9833) else {})) for label,w in [('eq',np.ones(n)),('nat',weights)]}
                    if a.self_check: assert all(x['delta']==0 and x['ci95']==[0,0] for x in stats['paired'][f].values())
                result[policy][h]=stats
        report['conditions'][cn]=result
    if skipped:
        report['skipped_queries']={'queries':skipped,'reason':'no stored c6-real condition to assert against (p2 route); not computed'}
    gates={}
    c5=report['conditions']['C5']['none']['all']['models']
    gates['C5_pair_flip']={'baseline':c5['p2']['frozen_C5_summary']['pair_flip_accuracy'],'lora':c5['lora']['frozen_C5_summary']['pair_flip_accuracy']}
    gates['C5_pair_flip']['passes']=gates['C5_pair_flip']['lora']>=gates['C5_pair_flip']['baseline']-1e-12
    for cn in ['sector_K8_center','noise_sigma10']:
        for policy,halves_ in report['conditions'][cn].items():
            ms=halves_['all']['models']; b=ms['p2']['only_interferer_nat']; l=ms['lora']['only_interferer_nat']
            gates[cn+'_'+policy]={'baseline':b,'lora':l,'delta':l-b,'passes':l<=b+.01+1e-12}
    report['gates']=gates
    report['remaining_gates_pass']=all(g['passes'] for g in gates.values())
    report['overall_section6_pass']=False
    report['overall_note']='C3 gate already failed; this analysis never changes that decision.'
    report['raw_sha256']={str(p/(('P2' if a.route_kind == 'audio_vs_t2' and m == 'p2' else raw_name) + '/raw.jsonl')):hashlib.sha256((p/(('P2' if a.route_kind == 'audio_vs_t2' and m == 'p2' else raw_name) + '/raw.jsonl')).read_bytes()).hexdigest() for m,ps in paths.items() for p in ps}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('INPUTS_AND_FROZEN_METRICS_PASS', flush=True)
    print(json.dumps(gates,indent=2),flush=True)


if __name__=='__main__':
    main()
