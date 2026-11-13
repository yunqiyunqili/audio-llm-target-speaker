"""CPU-only six-condition input audit. Run from the shared repository, never node-local /tmp."""
import runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for items,half,pattern in [
 ('devel_clean','selection','outputs/lora_base_audio/whisper_t2_devel_clean/*/W/raw.jsonl'),
 ('devel_clean','report','outputs/lora_base_audio/whisper_t2_devel_clean/*/W/raw.jsonl'),
 ('devel_rev','selection','outputs/lora_base_audio/whisper_t2_devel_rev/*/W/raw.jsonl'),
 ('devel_rev','report','outputs/lora_base_audio/whisper_t2_devel_rev/*/W/raw.jsonl'),
 ('final_selected','all','outputs/qwen_omni_final/22029639_selected/W/raw.jsonl'),
 ('final_other','all','outputs/qwen_omni_final/22029640_other/W/raw.jsonl')]:
 paths=sorted(root.glob(pattern));assert paths
 sys.argv=['eval_step2_lora.py','--run-dir','<SHARED_ROOT>/checkpoints/lora_v2/step2/lr5e-5','--step','220','--items',items,'--half',half,'--out','/tmp/not_written_step2_dryrun','--dry-run','--asr',*[str(p) for p in paths]]
 runpy.run_path(str(root/'scripts/lora/eval_step2_lora.py'),run_name='__main__')
print('ALL_SIX_INFERENCE_DRYRUNS_PASS',flush=True)
