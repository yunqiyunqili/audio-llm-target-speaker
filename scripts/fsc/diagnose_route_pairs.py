"""Read-only pair/source diagnostic for the frozen FSC route manifest."""
import json, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'scripts/sms_wsj'))
from eval_frontend_mix_v1_v3 import polarity_fix
from eval_frontend_mix_v1 import wrap

O=ROOT/'outputs/fsc_pilot'; R=O/'route_manifest'; MIX=O/'spatial_audio_real'; SEP=O/'spatialnet_ep100'; OUT=R/'pair_diagnostic.json'
def corr(a,b):
 n=min(len(a),len(b)); a,b=a[:n],b[:n]
 return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20))
def main():
 routes=[json.loads(x) for x in open(R/'route_manifest.jsonl')]
 mix={f"g{x['group']:03d}_s{x['swap']}":x for x in map(json.loads,open(MIX/'manifest.jsonl'))}
 assert len(routes)==64 and len(mix)==64
 rows=[]
 for r in routes:
  key=r['sample_id']; m=mix[key]
  obs,sr=sf.read(MIX/key/'obs_6ch_8k.wav',dtype='float64',always_2d=True); assert sr==8000
  y=np.stack([sf.read(SEP/key/f'out{k}.wav',dtype='float64')[0] for k in (0,1)])
  yf,_=polarity_fix(y,obs[:,0])
  refs=[sf.read(MIX/key/f'early_{s}_6ch_8k.wav',dtype='float64',always_2d=True)[0][:,0] for s in 'AB']
  C=np.array([[corr(yf[k],refs[j]) for j in (0,1)] for k in (0,1)])
  # Source assignment is post-hoc only: choose the permutation with larger sum.
  p=(0,1) if C[0,0]+C[1,1] >= C[0,1]+C[1,0] else (1,0)
  selected_source='AB'[p[r['selected_stream']]]; target_source=m['target']
  legacy_target_stream=int(np.argmax([C[k][0 if target_source=='A' else 1] for k in (0,1)]))
  rows.append(dict(sample_id=key,pair_id=f"g{m['group']:03d}",swap=m['swap'],target_source=target_source,
                   selected_stream=r['selected_stream'],source_by_stream=['AB'[p[0]],'AB'[p[1]]],
                   selected_source=selected_source,target_direction_deg=r['target_direction_deg'],
                   doa_deg=r['doa_deg'],selected_doa_error_deg=abs(float(wrap(r['doa_deg'][r['selected_stream']]-r['target_direction_deg']))),
                   source_correct=selected_source==target_source,correlation_matrix=C.tolist()))
  rows[-1]['legacy_target_stream']=legacy_target_stream
  rows[-1]['legacy_route_correct']=bool(legacy_target_stream==r['selected_stream'])
 by={}
 for x in rows: by.setdefault(x['pair_id'],[]).append(x)
 pairs=[]
 for pid,a in sorted(by.items()):
  assert len(a)==2
  ok=sum(x['source_correct'] for x in a)
  pairs.append(dict(pair_id=pid,rows=[x['sample_id'] for x in a],target_sources=[x['target_source'] for x in a],
                    selected_sources=[x['selected_source'] for x in a],selected_streams=[x['selected_stream'] for x in a],
                    both_correct=ok==2,one_correct=ok==1,both_wrong=ok==0,
                    target_identity_swap=(set(x['target_source'] for x in a)=={'A','B'}),
                    physical_position_swap=True,query_same=(a[0]['target_direction_deg']==a[1]['target_direction_deg']),
                    source_stream_permutation=[x['source_by_stream'] for x in a]))
 n=len(rows); wrong=[x for x in rows if not x['source_correct']]
 rep=dict(n=n,pairs=len(pairs),route_accuracy=sum(x['source_correct'] for x in rows)/n,
          legacy_route_accuracy=sum(x['legacy_route_correct'] for x in rows)/n,
          source_correct_counts={'both_correct':sum(x['both_correct'] for x in pairs),'one_correct':sum(x['one_correct'] for x in pairs),'both_wrong':sum(x['both_wrong'] for x in pairs)},
          wrong_samples=[x['sample_id'] for x in wrong],
          selected_doa_error_deg={'all_mean':float(np.mean([x['selected_doa_error_deg'] for x in rows])),'all_median':float(np.median([x['selected_doa_error_deg'] for x in rows])),'correct_mean':float(np.mean([x['selected_doa_error_deg'] for x in rows if x['source_correct']])), 'wrong_mean':float(np.mean([x['selected_doa_error_deg'] for x in wrong]))},
          legacy_wrong_samples=[x['sample_id'] for x in rows if not x['legacy_route_correct']],
          tau16_nearest_coverage=1.0, # inherited frozen summary; this is DOA proximity only
          interpretation='stream indices are not source identities; source assignment is post-hoc reference matching')
 (OUT).write_text(json.dumps({'summary':rep,'pairs':pairs,'rows':rows},indent=2)+'\n')
 print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
