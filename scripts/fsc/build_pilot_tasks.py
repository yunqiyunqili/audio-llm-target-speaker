"""Freeze the 256 FSC pilot tasks (CPU, before predictions)."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; O=ROOT/'outputs/fsc_pilot'; OUT=O/'pilot_tasks.jsonl'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 M=[json.loads(x) for x in open(O/'valid64_manifest.jsonl')]; overlap={x['pilot_index']:x for x in (json.loads(y) for y in open(O/'valid64_overlap.jsonl'))}
 routes={x['sample_id']:x for x in map(json.loads,open(O/'route_manifest/route_manifest.jsonl'))}
 uv=[json.loads(x) for x in open(O/'route_manifest/direction_uncertainty_manifest.jsonl') if json.loads(x)['query_delta_deg']!=0]
 # Exactly 16 samples per nonzero query offset, fixed by pilot index modulo 4.
 uv={(x['sample_id'],x['query_delta_deg']):x for x in uv}
 rows=[]
 fsc=ROOT/'data/fsc/source/fluent_speech_commands_dataset'
 for m in M:
  i=m['pilot_index']; base={'pilot_index':i,'target':m['target'],'transcript':m['transcript'],'speaker_id':m['speaker_id'],'overlap':overlap.get(i,{}),'source_manifest_sha256':m['audio_sha256']}
  clean=(fsc/m['path']).resolve(); rev=(O/'reverb16k'/f'{i:03d}.wav').resolve()
  rows += [dict(base,condition='clean',audio_path=str(clean),audio_sha256=sha(clean),query_delta_deg=0),dict(base,condition='reverb',audio_path=str(rev),audio_sha256=sha(rev),query_delta_deg=0)]
  r=routes[f'g{i//2:03d}_s{1 if i%2==0 else 2}']
  sign=-1.0 if r['flipped'][r['selected_stream']] else 1.0
  rows.append(dict(base,condition='spatial',audio_path=r['selected_audio_path'],audio_sha256=r['output_audio_sha256'],sample_id=r['sample_id'],selected_stream=r['selected_stream'],polarity_sign=sign,preprocess='chain8',query_delta_deg=0))
  delta=(-20.0,-10.0,10.0,20.0)[i%4]; u=uv[(r['sample_id'],delta)]
  rows.append(dict(base,condition='direction_uncertainty',audio_path=u['selected_audio_path'],audio_sha256=u['output_audio_sha256'],sample_id=u['sample_id'],selected_stream=u['selected_stream'],polarity_sign=sign,preprocess='chain8',query_delta_deg=delta,query_direction_deg=u['query_direction_deg'],within_tau=u['within_tau']))
 assert len(rows)==256 and all(os.path.exists(x['audio_path']) for x in rows)
 assert len({(x['condition'],x['pilot_index']) for x in rows})==256
 with open(OUT,'w') as f:
  for x in rows:f.write(json.dumps(x,sort_keys=True)+'\n')
 print(json.dumps({'n':len(rows),'by_condition':{c:sum(x['condition']==c for x in rows) for c in ('clean','reverb','spatial','direction_uncertainty')},'sha256':sha(OUT)},indent=2))
if __name__=='__main__':main()
