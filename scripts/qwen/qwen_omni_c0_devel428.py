"""C0 devel-428 baseline (docs/lora_slu_plan_v2.md §5, §11-4): SLURP reference transcripts of ALL 428 split_v2 devel recordings
-> frozen T2, text only, talker off. Same code path as qwen_omni_c0.py (C0 v1); only the item list differs.
Input format identical to C7W T2: build(processor, [sentence.strip() + '\\n', PROMPT_T2]). Greedy, thinker_max_new_tokens 256.
Records commit, dirty dependency files, sha256, node, GPU model, per-item seconds.
--dry-run N: CPU only, processor only (no model weights); builds all 124 inputs and prints N.
Usage: python scripts/qwen/qwen_omni_c0_devel428.py --out DIR [--dry-run N]
SLURP test (2026-11-12, docs/slurp_test_freeze_v1.md §0-4/§3): --split test [--part K --nparts N] [--adapter DIR]: the 1,183
  split_v2 test recordings (sorted by file), same input/decoding; --adapter loads the frozen T2-LoRA exactly as
  scripts/lora/eval_step2_lora.py (descriptive C0 + T2-LoRA). Devel default unchanged.
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

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import REVISION  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_T2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from slu_normalize import parse  # noqa: E402

DEPS = ['scripts/qwen/qwen_omni_c0_devel428.py', 'scripts/qwen/qwen_omni_c0_devel428.sbatch', 'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_single.py', 'data/slurp/manifests/split_v2_seed2026/devel_recordings.csv', 'docs/lora_slu_plan_v2.md']


def items(split='devel'):
    import csv
    rows = list(csv.DictReader(open(PROJ / f'data/slurp/manifests/split_v2_seed2026/{split}_recordings.csv', newline='')))
    N = dict(devel=428, test=1183)[split]
    assert len(rows) == N and len({r['file'] for r in rows}) == N
    return [dict(file=r['file'], slurp_id=int(r['slurp_id']), sentence=r['sentence'], intent=r['intent'], entities=json.loads(r['values']))
            for r in sorted(rows, key=lambda r: r['file'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--dry-run', type=int, default=0)
    ap.add_argument('--split', choices=('devel', 'test'), default='devel'); ap.add_argument('--adapter', type=Path)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    assert a.split == 'test' or (a.adapter is None and a.nparts == 1)
    from transformers import Qwen2_5OmniProcessor
    IT = items(a.split); n_all = len(IT); IT = IT[a.part * n_all // a.nparts:(a.part + 1) * n_all // a.nparts]
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS})
    if a.split == 'test':
        prov.update(split='test', part=a.part, nparts=a.nparts, n_set=n_all, adapter=str(a.adapter) if a.adapter else None)
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    B = {it['file']: build(qp, [it['sentence'].strip() + '\n', PROMPT_T2]) for it in IT}
    print('INPUTS_BUILT', len(B), flush=True)
    if a.dry_run:
        for it in IT[:a.dry_run]:
            t, b = B[it['file']]; print('DRY', it['file'], list(b['input_ids'].shape), repr(t[-160:]), flush=True)
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True)
        return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=False)
    prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(spec='docs/lora_slu_plan_v2.md (C0 devel-428 baseline)', model_commit=REVISION, prompt_T2=PROMPT_T2, n=len(IT),
                                                       decoding=dict(do_sample=False, thinker_max_new_tokens=256), **prov), indent=1, ensure_ascii=False) + '\n')
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    cond = 'C0'
    if a.adapter:  # identical to scripts/lora/eval_step2_lora.py
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        pm = get_peft_model(model.thinker, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)', bias='none'))
        pm.eval(); model.thinker.config.use_cache = True
        res = set_peft_model_state_dict(pm, load_file(str(a.adapter / 'adapter_model.safetensors')))
        assert not [k for k in getattr(res, 'unexpected_keys', []) if 'lora' in k], res
        bsum = float(sum(q.detach().double().abs().sum() for nm, q in pm.named_parameters() if 'lora_B' in nm)); assert bsum > 0
        print('ADAPTER', str(a.adapter), 'lora_B_abs_sum', round(bsum, 4), flush=True); cond = 'C0_T2LoRA'
        (a.out / 'adapter.json').write_text(json.dumps(dict(adapter=str(a.adapter), lora_B_abs_sum=bsum,
                                                            adapter_sha256=hashlib.sha256((a.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest())) + '\n')
    print('LOAD', round(time.perf_counter() - t0, 1), flush=True)
    d = a.out / cond; d.mkdir(); tc = time.perf_counter()
    with open(d / 'raw.jsonl', 'w') as fh:
        for it in IT:
            text, batch = B[it['file']]; batch = batch.to(model.device)
            torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); sec = time.perf_counter() - s0
            raw = qp.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(condition=cond, file=it['file'], slurp_id=it['slurp_id'], ref_sentence=it['sentence'], ref_intent=it['intent'],
                                     ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                                     new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1])), ensure_ascii=False) + '\n'); fh.flush()
    print('COND_DONE', cond, round(time.perf_counter() - tc, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
