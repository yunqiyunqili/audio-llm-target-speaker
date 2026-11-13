"""Frozen CPU comparison of Audio-LoRA and Whisper+T2-LoRA outputs."""
import argparse, json, collections, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/qwen'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/slurp'))
from binding_score_v1 import judge, vals
from rescore_slu_v2 import score_item

def rows(p): return {json.loads(x)['file']: json.loads(x) for x in open(p)}
def hit(r):
    return int(judge(vals(r['ref_entities']), collections.Counter(), vals(score_item(r)['hyp']['entities']))['t_any'])
def boot(d):
    by=collections.defaultdict(list)
    for k,v in d.items(): by[v[0]].append(v[1])
    s=np.array([sum(v) for v in by.values()]); c=np.array([len(v) for v in by.values()]); rng=np.random.default_rng(2026); ix=rng.integers(0,len(s),(2000,len(s))); z=s[ix].sum(1)/c[ix].sum(1)
    return {'delta':float(s.sum()/c.sum()),'ci95':[float(x) for x in np.quantile(z,[.025,.975])],'B':2000,'seed':2026}
def main(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False); report={}
    for name,ap,bp in [('C1_report',a.c1a,a.c1t),('C1R_report',a.c1ra,a.c1rt)]:
        A,T=rows(ap),rows(bp); T={k:v for k,v in T.items() if k in A}; assert set(A)==set(T)
        d={k:(A[k]['slurp_id'],hit(A[k])-hit(T[k])) for k in A}; report[name]={'n':len(A),'audio_any':sum(hit(x) for x in A.values())/len(A),'t2_any':sum(hit(x) for x in T.values())/len(T),'paired_audio_minus_t2':boot(d)}
    # C5 disagreement evidence; final-check JSON is the authoritative C5/C6 statistics.
    A={**rows(a.c5as),**rows(a.c5ao)}; T={**rows(a.c5ts),**rows(a.c5to)}; assert set(A)==set(T)=={f'g{g:03d}_s{s}' for g in range(100) for s in (1,2)}
    dis=[]
    for k in sorted(A):
        ha,ht=hit(A[k]),hit(T[k]);
        if ha!=ht or A[k]['raw_output'] != T[k]['raw_output']:
            cats=[]
            if T[k].get('asr_text','').strip().lower()!=A[k]['ref_sentence'].strip().lower(): cats.append('Whisper转写/输入差异证据')
            if A[k].get('parse_status')!='ok' or T[k].get('parse_status')!='ok': cats.append('解析/拒答')
            # P2 stores the physical stream while T2 is evaluated after frozen route selection;
            # absence of a T2 stream field is not itself a route disagreement.
            cats.append('Audio LLM理解/实体绑定差异')
            dis.append({'file':k,'audio_target_any':ha,'t2_target_any':ht,'categories':cats,'audio_raw':A[k]['raw_output'],'t2_raw':T[k]['raw_output'],'asr_text':T[k].get('asr_text'),'ref_sentence':A[k]['ref_sentence']})
    report['C5_disagreements']={'n':len(dis),'categories':dict(collections.Counter(c for x in dis for c in x['categories']))}
    (out/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with open(out/'disagreements.jsonl','w') as f:
      for x in dis:f.write(json.dumps(x,ensure_ascii=False)+'\n')
if __name__=='__main__':
  p=argparse.ArgumentParser();
  for n in ['c1a','c1t','c1ra','c1rt','c5as','c5ao','c5ts','c5to','out']: p.add_argument('--'+n,required=True)
  main(p.parse_args())
