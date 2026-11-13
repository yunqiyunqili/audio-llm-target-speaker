"""FSC extension v1 inference: one route x one task-list part per job (docs/fsc_extension_plan_v1.md §4-5).
Same model / prompts / decoding / preprocessing as the SLURP test and the FSC pilot:
  base       : audio -> pilot_audio.prepare() -> frozen P2, original Qwen2.5-Omni-7B (envs/qwen_omni)
  audio_lora : same + step1/lr1e-4/ckpt_step440 (envs/qwen_omni_lora)
  whisper    : audio -> prepare() -> Whisper large-v3 06f233fe (en, greedy, 128) -> asr
  wt2        : asr (from --asr) -> [asr.strip() + newline, PROMPT_T2], original model
  t2_lora    : same + step2/lr1e-4/ckpt_step660
  c0_t2 / c0_t2lora : gold transcription -> T2 (original / step660)
LoRA loading identical to scripts/lora/eval_step2_lora.py (r16/a32, q/k/v/o, use_cache, lora_B_abs_sum > 0).
Output OUT/predictions.jsonl (item_id, condition, route, raw_output, asr, audio/processed sha256, prompt) + OUT/config.json.
Usage: python scripts/fsc/ext_batch.py --route R --tasks TASKS.jsonl --out DIR [--asr W.jsonl ...] [--part K --nparts N]
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/qwen')); sys.path.insert(0, str(ROOT / 'scripts/fsc'))
from qwen_omni_c6c7 import build  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2  # noqa: E402
from qwen_omni_single import REVISION, SR  # noqa: E402
from qwen_omni_c7w_f1 import WHISPER_REV  # noqa: E402
from pilot_audio import prepare  # noqa: E402

SH = Path('<SHARED_ROOT>')
ADAPTER = dict(audio_lora=SH / 'checkpoints/lora_v2/step1/lr1e-4/ckpt_step440', t2_lora=SH / 'checkpoints/lora_v2/step2/lr1e-4/ckpt_step660',
               c0_t2lora=SH / 'checkpoints/lora_v2/step2/lr1e-4/ckpt_step660')
TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', choices=('base', 'audio_lora', 'whisper', 'wt2', 't2_lora', 'c0_t2', 'c0_t2lora'), required=True)
    ap.add_argument('--tasks', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--asr', type=Path, nargs='*', default=[])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    T = [json.loads(x) for x in open(a.tasks)]; n = len(T); assert len({t['item_id'] for t in T}) == n
    T = T[a.part * n // a.nparts:(a.part + 1) * n // a.nparts]
    text_in = a.route in ('wt2', 't2_lora', 'c0_t2', 'c0_t2lora'); audio_in = a.route in ('base', 'audio_lora', 'whisper')
    ASR = {}
    if a.route in ('wt2', 't2_lora'):
        for p in a.asr:
            for line in open(p):
                r = json.loads(line); assert r['route'] == 'whisper'; ASR[r['item_id']] = r['asr']
        assert all(t['item_id'] in ASR for t in T)
    git = lambda *x: subprocess.run(['git', '-C', str(ROOT), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(route=a.route, tasks=str(a.tasks), tasks_sha256=sha(a.tasks), part=a.part, nparts=a.nparts, n_part=len(T), n_set=n, git_commit=git('rev-parse', 'HEAD'),
                git_dirty=git('status', '--porcelain').splitlines(), node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                model_revision=REVISION, whisper_revision=WHISPER_REV, prompt='T2' if text_in else ('P2' if a.route != 'whisper' else None),
                asr_sha256={str(p): sha(p) for p in a.asr}, adapter=str(ADAPTER.get(a.route)) if a.route in ADAPTER else None)
    if a.route in ADAPTER:
        prov['adapter_sha256'] = sha(ADAPTER[a.route] / 'adapter_model.safetensors')
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration, WhisperProcessor, WhisperForConditionalGeneration
    hub = Path(os.environ['HF_HOME']) / 'hub'
    qdir = hub / 'models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION; wdir = hub / 'models--openai--whisper-large-v3/snapshots' / WHISPER_REV
    t0 = time.perf_counter()
    if a.route == 'whisper':
        wm = WhisperForConditionalGeneration.from_pretrained(str(wdir), torch_dtype=torch.float16, local_files_only=True).to('cuda').eval()
        wp = WhisperProcessor.from_pretrained(str(wdir), local_files_only=True)
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
        model.disable_talker(); qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
        if a.route in ADAPTER:
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
            from safetensors.torch import load_file
            pm = get_peft_model(model.thinker, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_RE, bias='none'))
            pm.eval(); model.thinker.config.use_cache = True
            res = set_peft_model_state_dict(pm, load_file(str(ADAPTER[a.route] / 'adapter_model.safetensors')))
            assert not [k for k in getattr(res, 'unexpected_keys', []) if 'lora' in k], res
            prov['lora_B_abs_sum'] = float(sum(q.detach().double().abs().sum() for nm, q in pm.named_parameters() if 'lora_B' in nm)); assert prov['lora_B_abs_sum'] > 0
    prov['gpu'] = torch.cuda.get_device_name(0); prov['load_seconds'] = time.perf_counter() - t0
    a.out.mkdir(parents=True, exist_ok=False); (a.out / 'config.json').write_text(json.dumps(prov, indent=1) + '\n'); print('PROVENANCE', json.dumps(prov), flush=True)
    tc = time.perf_counter()
    with open(a.out / 'predictions.jsonl', 'w') as fh:
        for it in T:
            rec = dict(item_id=it['item_id'], condition=it['condition'], route=a.route, stream_role=it.get('stream_role'))
            if audio_in:
                assert sha(it['audio_path']) == it['audio_sha256']
                wav, pinfo = prepare(it); rec.update(audio_sha256=it['audio_sha256'], processed_waveform_sha256=hashlib.sha256(wav.tobytes()).hexdigest(), n_chain8=pinfo['n_chain8'])
            s0 = time.perf_counter()
            if a.route == 'whisper':
                f = wp(wav, sampling_rate=SR, return_tensors='pt').input_features.to('cuda', torch.float16)
                with torch.inference_mode():
                    ids = wm.generate(f, language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128)
                rec.update(asr=wp.batch_decode(ids, skip_special_tokens=True)[0], raw_output=None)
            else:
                if text_in:
                    src = ASR[it['item_id']] if a.route in ('wt2', 't2_lora') else it['text']
                    text, b = build(qp, [src.strip() + '\n', PROMPT_T2]); rec['asr'] = ASR.get(it['item_id'])
                else:
                    text, b = build(qp, PROMPT_P2, wav)
                for k, v in b.items():
                    if torch.is_floating_point(v):
                        b[k] = v.to(model.dtype)
                with torch.inference_mode():
                    ids = model.generate(**{k: v.to(model.device) for k, v in b.items()}, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
                rec.update(raw_output=qp.batch_decode(ids[:, b['input_ids'].shape[1]:], skip_special_tokens=True)[0], prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
            rec['seconds'] = time.perf_counter() - s0
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
    print('EXT_DONE', a.route, len(T), round(time.perf_counter() - tc, 1), flush=True)


if __name__ == '__main__':
    main()
