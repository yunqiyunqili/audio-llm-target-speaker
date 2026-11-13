"""Create the frozen FSC pilot spatial_list.json (reverb moved to make_reverb16k.py on 2026-11-11)."""
import json, hashlib
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT=Path(__file__).resolve().parents[2]; FSC=ROOT/'data/fsc/source/fluent_speech_commands_dataset'; O=ROOT/'outputs/fsc_pilot'
M=[json.loads(x) for x in open(O/'valid64_manifest.jsonl')]
L=json.load(open(ROOT/'configs/mix_v1_devel100_v2.json'))['groups'][:32]
sp=O/'spatial_list.json'; groups=[]
for i,g in enumerate(L):
    a,b=M[2*i],M[2*i+1]
    def info(x):
        y,sr=sf.read(FSC/x['path'],dtype='float64'); assert sr==16000 and y.ndim==1
        return {'file':str((FSC/x['path']).resolve()),'n_16k':len(y),'n_8k':len(resample_poly(y,1,2)),'offset_8k':0}
    aa,bb=info(a),info(b); T=max(aa['n_8k'],bb['n_8k'])
    groups.append({'group':i,'T_8k':T,'A':aa,'B':bb,'rir':g['rir'],'noise':{'snr_db':25.0,'seed':202600+i},'swaps':[{'swap':1,'target':'A','A_pos':0,'B_pos':1,'offset_8k':0},{'swap':2,'target':'B','A_pos':1,'B_pos':0,'offset_8k':0}]})
sp.write_text(json.dumps({'groups':groups,'source_manifest_sha256':hashlib.sha256((O/'valid64_manifest.jsonl').read_bytes()).hexdigest(),'seed':2026},indent=2)+'\n')
# 2026-11-11: the single-speaker reverb that used to be generated here was wrong (16 kHz speech convolved with the
# 8 kHz RIR, then upsampled again; no rir_start alignment; peak instead of -30 dBFS level). It is void.
# Reverb is now produced by scripts/fsc/make_reverb16k.py, which calls scripts/lora/reverb_single.reverb() (= SLURP C1-R).
print('WROTE',sp,len(M))
