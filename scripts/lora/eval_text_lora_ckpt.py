"""Step-0 checkpoint inference on devel reference transcripts (C0 口径; docs/lora_slu_plan_v2.md + revision 1).
Loads the base model once, then for each checkpoint loads the LoRA adapter weights and generates (frozen T2, greedy,
thinker_max_new_tokens 256, identical to C0). Generates once per unique slurp_id of the requested half and maps the output to
every recording of that sentence (C0-428 showed identical outputs for duplicate sentences). Output format = C0-428 raw.jsonl.
--dry-run: CPU, tokenizer only.
Usage: python scripts/lora/eval_text_lora_ckpt.py --run-dir DIR --steps 74,148 --half {selection,report,all} --out DIR [--dry-run]
"""
import argparse
import csv
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen'))
from qwen_omni_single import REVISION  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_T2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from slu_normalize import parse  # noqa: E402

TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'


def devel_half(half):
    rows = list(csv.DictReader(open(PROJ / 'data/slurp/manifests/split_v2_seed2026/devel_recordings.csv', newline='')))
    ids = sorted({int(r['slurp_id']) for r in rows}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    keep = {'selection': lambda i: i in sel, 'report': lambda i: i not in sel, 'all': lambda i: True}[half]
    return sorted([r for r in rows if keep(int(r['slurp_id']))], key=lambda r: r['file'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--steps', required=True)
    ap.add_argument('--half', choices=('selection', 'report', 'all'), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(',')]
    from transformers import Qwen2_5OmniProcessor
    rows = devel_half(a.half); uniq = {}
    for r in rows:
        uniq.setdefault(int(r['slurp_id']), r['sentence'])
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    B = {i: build(qp, [s.strip() + '\n', PROMPT_T2]) for i, s in uniq.items()}
    for st in steps:
        assert (a.run_dir / f'ckpt_step{st}' / 'adapter_model.safetensors').exists() or a.dry_run, st
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', 'scripts/lora', 'scripts/qwen').splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'), run_dir=str(a.run_dir), steps=steps, half=a.half)
    print('ITEMS', len(rows), 'unique', len(uniq), flush=True)
    if a.dry_run:
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen2_5OmniForConditionalGeneration
    prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    pm = get_peft_model(model.thinker, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_RE, bias='none'))
    pm.eval(); model.thinker.config.use_cache = True
    for st in steps:
        sd = load_file(str(a.run_dir / f'ckpt_step{st}' / 'adapter_model.safetensors'))
        res = set_peft_model_state_dict(pm, sd)
        assert not [k for k in getattr(res, 'unexpected_keys', []) if 'lora' in k], res
        bsum = float(sum(q.detach().double().abs().sum() for n, q in pm.named_parameters() if 'lora_B' in n)); assert bsum > 0, 'adapter not loaded'
        print('ADAPTER', st, 'lora_B_abs_sum', round(bsum, 4), flush=True)
        d = a.out / f'step{st}' / a.half; d.mkdir(parents=True, exist_ok=False)
        (d / 'config.json').write_text(json.dumps(dict(prompt_T2=PROMPT_T2, step=st, n_recordings=len(rows), n_unique=len(uniq), **prov), indent=1, ensure_ascii=False) + '\n')
        gen = {}; tc = time.perf_counter()
        for i, (text, b) in B.items():
            b = b.to(model.device); torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**b, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize()
            gen[i] = (text, qp.batch_decode(ids[:, b['input_ids'].shape[1]:], skip_special_tokens=True)[0], time.perf_counter() - s0, int(ids.shape[1] - b['input_ids'].shape[1]))
        with open(d / 'raw.jsonl', 'w') as fh:
            for r in rows:
                text, raw, sec, nt = gen[int(r['slurp_id'])]
                fh.write(json.dumps(dict(condition=f'T0LoRA_step{st}', file=r['file'], slurp_id=int(r['slurp_id']), ref_sentence=r['sentence'], ref_intent=r['intent'],
                                         ref_entities=json.loads(r['values']), prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                                         new_tokens=nt), ensure_ascii=False) + '\n')
        print('CKPT_DONE', st, round(time.perf_counter() - tc, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
