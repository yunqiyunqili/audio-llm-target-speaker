# DEPRECATED (2026-11-11): do not use. The P2 route below applies chain8 to 16 kHz clean/reverb audio and a second
# time to spatial audio, which does not match the SLURP C1 / C1-R / C5 chains (docs/fsc_pilot_plan.md, 2026-11-11 fix).
# Smoke jobs 22057002 / 22057022 / 22057024 that used it are void. Use scripts/fsc/qwen_pilot_batch.py --limit N instead.
import sys as _sys; _sys.exit('qwen_smoke.py is deprecated; use scripts/fsc/qwen_pilot_batch.py --limit N')
import argparse, json, os, sys
from pathlib import Path
import soundfile as sf
PROJ=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJ/'scripts/qwen'))
from qwen_omni_c6c7 import build
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2
from qwen_omni_single import REVISION, SR
from qwen_omni_c7w_f1 import WHISPER_REV, chain8
ap=argparse.ArgumentParser()
ap.add_argument('--route',choices=['base','audio_lora','t2_lora'],required=True)
ap.add_argument('--audio',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
import torch
from transformers import Qwen2_5OmniProcessor,Qwen2_5OmniForConditionalGeneration,WhisperProcessor,WhisperForConditionalGeneration
qdir=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots'/REVISION
wdir=Path(os.environ['HF_HOME'])/'hub/models--openai--whisper-large-v3/snapshots'/WHISPER_REV
wav,sr=sf.read(a.audio,dtype='float32'); assert sr==16000 and wav.ndim==1
model=Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir),torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
model.disable_talker(); qp=Qwen2_5OmniProcessor.from_pretrained(str(qdir),local_files_only=True)
adapter=None
if a.route=='audio_lora': adapter='<SHARED_ROOT>/checkpoints/lora_v2/step1/lr1e-4/ckpt_step440'
if a.route=='t2_lora': adapter='<SHARED_ROOT>/checkpoints/lora_v2/step2/lr1e-4/ckpt_step660'
if adapter:
 from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
 from safetensors.torch import load_file
 pm=get_peft_model(model.thinker,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',bias='none'))
 pm.eval(); set_peft_model_state_dict(pm,load_file(str(Path(adapter)/'adapter_model.safetensors')))
if a.route=='t2_lora':
 wm=WhisperForConditionalGeneration.from_pretrained(str(wdir),torch_dtype=torch.float16).to('cuda').eval()
 wp=WhisperProcessor.from_pretrained(str(wdir),local_files_only=True)
 feat=wp(wav,sampling_rate=SR,return_tensors='pt').input_features.to('cuda',torch.float16)
 with torch.inference_mode(): ids=wm.generate(feat,language='en',task='transcribe',num_beams=1,do_sample=False,max_new_tokens=128)
 asr=wp.batch_decode(ids,skip_special_tokens=True)[0]; text,batch=build(qp,[asr.strip()+'\n',PROMPT_T2])
else:
 asr=None; text,batch=build(qp,PROMPT_P2,chain8(wav))
for k,v in batch.items():
 if torch.is_floating_point(v): batch[k]=v.to(model.dtype)
with torch.inference_mode(): ids=model.generate(**{k:v.to(model.device) for k,v in batch.items()},return_audio=False,thinker_max_new_tokens=256,do_sample=False,use_audio_in_video=False)
raw=qp.batch_decode(ids[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)[0]
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'route':a.route,'audio':str(a.audio),'asr':asr,'raw_output':raw,'checkpoint':adapter,'prompt':'T2' if a.route=='t2_lora' else 'P2'},indent=2)+'\n')
print('SMOKE_PASS',a.route,raw)
