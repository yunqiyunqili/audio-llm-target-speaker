"""Frozen SpatialNet ep100 inference on FSC pilot 6-channel inputs."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
ROOT=Path('<SEPARATION_REPO>/multichannel'); RUN=ROOT/'spatialnet/exp/sms_spatialnet_small_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
ap=argparse.ArgumentParser(); ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--limit',type=int,default=0); a=ap.parse_args()
sys.path.insert(0,str(ROOT/'spatialnet')); import torch, train as T
summary=json.load(open(RUN/'test_summary.json')); pkg=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False)
assert summary['selected_epoch']==100 and sha(RUN/'best.pt')==summary['checkpoint_sha256']
dev='cuda' if torch.cuda.is_available() else 'cpu'; model=T.Model(6).to(dev); model.load_state_dict(pkg['model']); model.eval()
dirs=sorted(p for p in a.inputs.iterdir() if p.is_dir() and (p/'obs_6ch_8k.wav').exists()); todo=dirs[:a.limit] if a.limit else dirs
a.out.mkdir(parents=True,exist_ok=False); rows=[]; t0=time.perf_counter()
with torch.no_grad():
 for d in todo:
  x,sr=sf.read(d/'obs_6ch_8k.wav',dtype='float32',always_2d=True); assert sr==8000 and x.shape[1]==6
  z=torch.from_numpy(x)[None].to(dev); xin,_,scale=T.normalize(z,z[...,:2].transpose(1,2)); y=(model(xin)*scale)[0].detach().cpu().numpy(); assert y.shape==(2,len(x)) and np.isfinite(y).all()
  od=a.out/d.name; od.mkdir()
  for k in range(2): sf.write(od/f'out{k}.wav',y[k].astype('float32'),8000,subtype='FLOAT')
  rows.append({'key':d.name,'samples':len(x),'seconds':float(len(x)/8000)})
with open(a.out/'manifest.jsonl','w') as f:
 for r in rows:f.write(json.dumps(r)+'\n')
(a.out/'summary.json').write_text(json.dumps({'epoch':100,'checkpoint_sha256':summary['checkpoint_sha256'],'device':dev,'n':len(rows),'elapsed':time.perf_counter()-t0,'slurm_job_id':os.getenv('SLURM_JOB_ID')},indent=2)+'\n')
print('SPATIALNET_DONE',len(rows))
