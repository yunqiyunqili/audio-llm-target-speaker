"""Downstream evaluation of a snapshot front-end's selected target stream (docs/frontend_compare_v1.md). IN-TRAINING snapshot.
Selection = OUT_EVAL/<model>/selection.jsonl (v3 rule, oracle query azimuth). Audio = selected stream, polarity applied,
C5 chain (resample_poly 2/1, -30 dBFS). P2 (frozen) and Whisper large-v3 (06f233fe..., en/transcribe, greedy, 128) + frozen T2.
Records commit, node, GPU model, per-item seconds. --dry-run N: CPU only.
Usage: python scripts/qwen/qwen_omni_snapshot.py --model M --sep-dir DIR --selection PATH --out DIR [--dry-run N]
SLURP test (2026-11-12, docs/slurp_test_freeze_v1.md): env SLURP_EVAL_SPLIT=test (samples() follows split_cfg.LIST) and
  --part K --nparts N (samples [K*n//N, (K+1)*n//N)); same chain, prompts and decoding. Devel default unchanged.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import REVISION, SR  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from qwen_omni_c7w_f1 import WHISPER_REV, chain8, samples  # noqa: E402
from slu_normalize import parse  # noqa: E402

DEPS = ['scripts/qwen/qwen_omni_snapshot.py', 'scripts/qwen/qwen_omni_snapshot.sbatch', 'scripts/qwen/qwen_omni_c7w_f1.py', 'scripts/qwen/qwen_omni_c6c7.py',
        'scripts/qwen/qwen_omni_c4.py', 'scripts/qwen/qwen_omni_c1_p2t2.py', 'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py',
        'scripts/sms_wsj/eval_snapshot_mix.py', 'docs/frontend_compare_v1.md']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--sep-dir', type=Path, required=True); ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--dry-run', type=int, default=0)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--adapter', type=Path, help='Fixed LoRA checkpoint directory; requires --p2-only')
    ap.add_argument('--p2-only', action='store_true')
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    assert a.adapter is None or a.p2_only
    from transformers import Qwen2_5OmniProcessor, WhisperProcessor
    S = samples(); sel = {r['key']: r for r in map(json.loads, open(a.selection))}
    nS = len(S); S = S[a.part * nS // a.nparts:(a.part + 1) * nS // a.nparts]
    A = {}
    for s in S:
        k = s['key']; st = sel[k]['stream']; sign = -1.0 if sel[k]['flipped'][st] else 1.0
        A[k] = chain8(sign * sf.read(a.sep_dir / k / f'out{st}.wav', dtype='float64')[0])
    a.out.mkdir(parents=True, exist_ok=False)
    hub = Path(os.environ['HF_HOME']) / 'hub'
    qdir = hub / 'models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION; wdir = hub / 'models--openai--whisper-large-v3/snapshots' / WHISPER_REV
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(model=a.model, git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={**{p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS}, str(a.selection): hashlib.sha256(a.selection.read_bytes()).hexdigest()},
                note=('FINAL SpatialNet; oracle query azimuth' if a.final else 'IN-TRAINING SNAPSHOT, diagnosis only; oracle query azimuth'),
                audio_sha256={k: hashlib.sha256(w.tobytes()).hexdigest() for k, w in A.items()},
                adapter=str(a.adapter) if a.adapter else None)
    if a.nparts > 1:
        prov.update(part=a.part, nparts=a.nparts, n_set=nS)
    import split_cfg
    if split_cfg.SPLIT != 'devel':
        prov.update(split=split_cfg.SPLIT, list=str(split_cfg.LIST))
    if a.adapter:
        prov['adapter_sha256'] = hashlib.sha256((a.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
    print('AUDIO_BUILT', len(A), 'peak', round(max(float(np.abs(w).max()) for w in A.values()), 3), flush=True)
    if a.dry_run:
        qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True); wp = WhisperProcessor.from_pretrained(str(wdir), local_files_only=True)
        for s in S[:a.dry_run]:
            t, b = build(qp, PROMPT_P2, A[s['key']]); f = wp(A[s['key']], sampling_rate=SR, return_tensors='pt').input_features
            print('DRY', s['key'], list(b['input_ids'].shape), list(f.shape), flush=True)
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True)
        return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, WhisperForConditionalGeneration
    prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker(); qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    if a.adapter:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        pm = get_peft_model(model.thinker, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)', bias='none'))
        pm.eval(); model.thinker.config.use_cache = True
        res = set_peft_model_state_dict(pm, load_file(str(a.adapter / 'adapter_model.safetensors')))
        assert not [k for k in getattr(res, 'unexpected_keys', []) if 'lora' in k], res
        prov['lora_B_abs_sum'] = float(sum(q.detach().double().abs().sum() for n, q in pm.named_parameters() if 'lora_B' in n))
        assert abs(prov['lora_B_abs_sum'] - 6594.5746) < 0.001
        print('ADAPTER', str(a.adapter), prov['lora_B_abs_sum'], flush=True)
    if not a.p2_only:
        wm = WhisperForConditionalGeneration.from_pretrained(str(wdir), torch_dtype=torch.float16, local_files_only=True).to('cuda').eval()
        wp = WhisperProcessor.from_pretrained(str(wdir), local_files_only=True)
    (a.out / 'config.json').write_text(json.dumps(dict(prompt_P2=PROMPT_P2, prompt_T2=PROMPT_T2, whisper_revision=WHISPER_REV, **prov), indent=1) + '\n')
    print('LOAD', round(time.perf_counter() - t0, 1), flush=True)

    def qgen(text, batch):
        batch = batch.to(model.device)
        for k, v in batch.items():
            if torch.is_floating_point(v):
                batch[k] = v.to(model.dtype)
        torch.cuda.synchronize(); s0 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
        torch.cuda.synchronize()
        return qp.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0], time.perf_counter() - s0, int(ids.shape[1] - batch['input_ids'].shape[1])

    def rec(name, s, raw, sec, nt, **extra):
        return dict(condition=name, file=s['key'], group=s['group'], swap=s['swap'], target=s['target'], slurp_id=None, ref_sentence=s['sentence'],
                    ref_intent=s['intent'], ref_entities=s['entities'], raw_output=raw, parse_status=parse(raw)[1], seconds=sec, new_tokens=nt, **extra)
    asr = {}
    for name in (('P2',) if a.p2_only else ('P2', 'W', 'T2')):
        d = a.out / name; d.mkdir(); tc = time.perf_counter()
        with open(d / 'raw.jsonl', 'w') as fh:
            for s in S:
                k = s['key']
                if name == 'P2':
                    t, b = build(qp, PROMPT_P2, A[k]); raw, sec, nt = qgen(t, b); r = rec('P2', s, raw, sec, nt, stream=sel[k]['stream'], prompt_text=t)
                elif name == 'W':
                    f = wp(A[k], sampling_rate=SR, return_tensors='pt').input_features.to('cuda', torch.float16)
                    torch.cuda.synchronize(); s0 = time.perf_counter()
                    with torch.inference_mode():
                        ids = wm.generate(f, language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128)
                    torch.cuda.synchronize(); asr[k] = wp.batch_decode(ids, skip_special_tokens=True)[0]
                    r = rec('W', s, asr[k], time.perf_counter() - s0, int(ids.shape[1]))
                else:
                    t, b = build(qp, [asr[k].strip() + '\n', PROMPT_T2]); raw, sec, nt = qgen(t, b); r = rec('T2', s, raw, sec, nt, asr_text=asr[k])
                fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
        print('COND_DONE', name, round(time.perf_counter() - tc, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
