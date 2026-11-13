"""Frozen FSC pilot inference: one model load, 256 task rows, JSONL outputs."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'scripts/qwen'))
from qwen_omni_c6c7 import build
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2
from qwen_omni_single import REVISION, SR
from qwen_omni_c7w_f1 import WHISPER_REV
sys.path.insert(0,str(ROOT/'scripts/fsc'))
from pilot_audio import prepare  # single preprocessing path per condition (C1 / C1-R / C5 chain8 once)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
ap=argparse.ArgumentParser(); ap.add_argument('--route',choices=['base','audio_lora','t2_lora'],required=True); ap.add_argument('--tasks',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--limit',type=int,default=0); a=ap.parse_args()
tasks=[json.loads(x) for x in open(a.tasks)]; assert len(tasks)==256 and len({(x['condition'],x['pilot_index']) for x in tasks})==256
if a.limit: tasks=tasks[:a.limit]
import torch
from transformers import Qwen2_5OmniProcessor,Qwen2_5OmniForConditionalGeneration,WhisperProcessor,WhisperForConditionalGeneration
qdir=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots'/REVISION; wdir=Path(os.environ['HF_HOME'])/'hub/models--openai--whisper-large-v3/snapshots'/WHISPER_REV
t0=time.perf_counter(); model=Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir),torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True); model.disable_talker(); qp=Qwen2_5OmniProcessor.from_pretrained(str(qdir),local_files_only=True); load=time.perf_counter()-t0
adapter=None
if a.route=='audio_lora': adapter='<SHARED_ROOT>/checkpoints/lora_v2/step1/lr1e-4/ckpt_step440'
if a.route=='t2_lora': adapter='<SHARED_ROOT>/checkpoints/lora_v2/step2/lr1e-4/ckpt_step660'
if adapter:
 from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
 from safetensors.torch import load_file
 pm=get_peft_model(model.thinker,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',bias='none')); pm.eval(); set_peft_model_state_dict(pm,load_file(str(Path(adapter)/'adapter_model.safetensors')))
 wm=wp=None
 if a.route=='t2_lora': wm=WhisperForConditionalGeneration.from_pretrained(str(wdir),torch_dtype=torch.float16).to('cuda').eval(); wp=WhisperProcessor.from_pretrained(str(wdir),local_files_only=True)
a.out.mkdir(parents=False,exist_ok=False); fout=a.out/'predictions.jsonl'; tstart=time.perf_counter(); by={}
with open(fout,'w') as f:
 for idx,it in enumerate(tasks):
  assert sha(it['audio_path'])==it['audio_sha256']
  wav,pinfo=prepare(it); actual_sr=16000
  if a.route=='t2_lora':
   feat=wp(wav,sampling_rate=SR,return_tensors='pt').input_features.to('cuda',torch.float16)
   with torch.inference_mode(): ids=wm.generate(feat,language='en',task='transcribe',num_beams=1,do_sample=False,max_new_tokens=128)
   asr=wp.batch_decode(ids,skip_special_tokens=True)[0]; text,batch=build(qp,[asr.strip()+'\n',PROMPT_T2])
  else: asr=None; text,batch=build(qp,PROMPT_P2,wav)
  for k,v in batch.items():
   if torch.is_floating_point(v): batch[k]=v.to(model.dtype)
  with torch.inference_mode(): ids=model.generate(**{k:v.to(model.device) for k,v in batch.items()},return_audio=False,thinker_max_new_tokens=256,do_sample=False,use_audio_in_video=False)
  raw=qp.batch_decode(ids[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)[0]
  rec=dict(index=idx,route=a.route,condition=it['condition'],pilot_index=it['pilot_index'],sample_id=it.get('sample_id'),query_delta_deg=it.get('query_delta_deg'),audio_path=it['audio_path'],audio_sha256=it['audio_sha256'],processed_waveform_sha256=hashlib.sha256(wav.tobytes()).hexdigest(),actual_input_sr=actual_sr,preprocess=it.get('preprocess','identity'),polarity_sign=it.get('polarity_sign',1.0),n_chain8=pinfo['n_chain8'],asr=asr,raw_output=raw,checkpoint=adapter,prompt='T2' if a.route=='t2_lora' else 'P2')
  f.write(json.dumps(rec,ensure_ascii=False,sort_keys=True)+'\n'); f.flush(); by[it['condition']]=by.get(it['condition'],0)+1
  if (idx+1)%16==0: print('DONE',idx+1,flush=True)
(a.out/'summary.json').write_text(json.dumps(dict(route=a.route,n=len(tasks),by_condition=by,load_seconds=load,wall_seconds=time.perf_counter()-tstart,tasks_sha256=sha(a.tasks),model_revision=REVISION,whisper_revision=WHISPER_REV,checkpoint=adapter),indent=2)+'\n')
print('PILOT_DONE',a.route,len(tasks),flush=True)
