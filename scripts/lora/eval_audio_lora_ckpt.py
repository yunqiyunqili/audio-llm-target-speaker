"""Step-1 audio LoRA checkpoint inference on devel audio (docs/lora_v2_runs.md §15, §17). ONE checkpoint per job.
Base model + LoRA adapter (same loading as eval_text_lora_ckpt.py), frozen P2, greedy, thinker_max_new_tokens 256 — the same
inference path as baseline_audio.py --mode p2 (same audio items, same build(), same float casting). Items: devel_clean (C1) or
devel_rev (C1-R), restricted to a half (selection / report / all; split by slurp_id, Random(2026), as in plan §3.2).
--dry-run: CPU, processor only.
Usage: python scripts/lora/eval_audio_lora_ckpt.py --run-dir DIR --step S --items {devel_clean,devel_rev} --half H --out DIR [--dry-run]
SLURP test (2026-11-12, docs/slurp_test_freeze_v1.md): --items test_clean|test_rev --half all [--part K --nparts N]; parts are
  items [K*n//N, (K+1)*n//N) of the sorted list and write to <items>_all_part<K>of<N>. Devel behaviour unchanged.
"""
import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/lora'))
from qwen_omni_single import REVISION  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from slu_normalize import parse  # noqa: E402
from baseline_audio import items, load_audio  # noqa: E402

TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--items', choices=('devel_clean', 'devel_rev', 'test_clean', 'test_rev'), required=True)
    ap.add_argument('--half', choices=('selection', 'report', 'all'), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    assert a.items.startswith('devel_') and a.nparts == 1 or a.items.startswith('test_') and a.half == 'all'
    IT = items(a.items); ids = sorted({it['slurp_id'] for it in IT}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    keep = {'selection': lambda i: i in sel, 'report': lambda i: i not in sel, 'all': lambda i: True}[a.half]
    IT = [it for it in IT if keep(it['slurp_id'])]
    n_all = len(IT); IT = IT[a.part * n_all // a.nparts:(a.part + 1) * n_all // a.nparts]
    W = {it['file']: load_audio(it, {}, False)[0] for it in IT}
    ck = a.run_dir / f'ckpt_step{a.step}' / 'adapter_model.safetensors'; assert ck.exists() or a.dry_run, ck
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', 'scripts/lora', 'scripts/qwen').splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'), run_dir=str(a.run_dir), step=a.step,
                items=a.items, half=a.half, n=len(IT), **(dict(part=a.part, nparts=a.nparts, n_set=n_all) if a.items.startswith('test_') else {}))
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    from transformers import Qwen2_5OmniProcessor
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    print('ITEMS', len(IT), a.items, a.half, flush=True)
    if a.dry_run:
        for it in IT[:2]:
            print('DRY', it['file'], list(build(qp, PROMPT_P2, W[it['file']])[1]['input_ids'].shape), flush=True)
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen2_5OmniForConditionalGeneration
    prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    pm = get_peft_model(model.thinker, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_RE, bias='none'))
    pm.eval(); model.thinker.config.use_cache = True
    res = set_peft_model_state_dict(pm, load_file(str(ck)))
    assert not [k for k in getattr(res, 'unexpected_keys', []) if 'lora' in k], res
    bsum = float(sum(q.detach().double().abs().sum() for n, q in pm.named_parameters() if 'lora_B' in n)); assert bsum > 0
    print('ADAPTER', a.step, 'lora_B_abs_sum', round(bsum, 4), 'load_s', round(time.perf_counter() - t0, 1), flush=True)
    d = a.out / f'step{a.step}' / (f'{a.items}_{a.half}' + (f'_part{a.part}of{a.nparts}' if a.nparts > 1 else '')); d.mkdir(parents=True, exist_ok=False)
    (d / 'config.json').write_text(json.dumps(dict(prompt_P2=PROMPT_P2, lora_B_abs_sum=bsum, **prov), indent=1, ensure_ascii=False) + '\n')
    tc = time.perf_counter()
    with open(d / 'raw.jsonl', 'w') as fh:
        for it in IT:
            text, b = build(qp, PROMPT_P2, W[it['file']]); b = b.to(model.device)
            for k, v in b.items():
                if torch.is_floating_point(v):
                    b[k] = v.to(model.dtype)
            torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**b, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); raw = qp.batch_decode(ids[:, b['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(condition=f'P2LoRA_step{a.step}', file=it['file'], audio=it['audio'], slurp_id=it['slurp_id'], ref_sentence=it['sentence'],
                                     ref_intent=it['intent'], ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1],
                                     seconds=time.perf_counter() - s0, new_tokens=int(ids.shape[1] - b['input_ids'].shape[1])), ensure_ascii=False) + '\n'); fh.flush()
    print('CKPT_DONE', a.step, round(time.perf_counter() - tc, 1), flush=True); print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
