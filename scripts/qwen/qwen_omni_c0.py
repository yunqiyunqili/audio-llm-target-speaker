"""C0 v1 (docs/c0_plan_v1.md): SLURP reference transcript of the 124 mix_v1 recordings -> frozen T2, text only, talker off.
Input format identical to C7W T2: build(processor, [sentence.strip() + '\\n', PROMPT_T2]). Greedy, thinker_max_new_tokens 256.
Records commit, dirty dependency files, sha256, node, GPU model, per-item seconds.
--dry-run N: CPU only, processor only (no model weights); builds all 124 inputs and prints N.
Usage: python scripts/qwen/qwen_omni_c0.py --out DIR [--dry-run N]
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

DEPS = ['scripts/qwen/qwen_omni_c0.py', 'scripts/qwen/qwen_omni_c0.sbatch', 'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_single.py', 'configs/mix_v1_devel100_v2.json', 'docs/c0_plan_v1.md']


def items():
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']
    rec = {}
    for g in L:
        for k in 'AB':
            r = g[k]
            if r['file'] in rec:
                assert rec[r['file']]['sentence'] == r['sentence']
            rec[r['file']] = dict(file=r['file'], sentence=r['sentence'], intent=r['intent'], entities=r['entities'])
    assert len(rec) == 124, len(rec)
    return [rec[f] for f in sorted(rec)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    IT = items()
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS})
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
    (a.out / 'config.json').write_text(json.dumps(dict(spec='docs/c0_plan_v1.md', model_commit=REVISION, prompt_T2=PROMPT_T2, n=len(IT),
                                                       decoding=dict(do_sample=False, thinker_max_new_tokens=256), **prov), indent=1, ensure_ascii=False) + '\n')
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    print('LOAD', round(time.perf_counter() - t0, 1), flush=True)
    d = a.out / 'C0'; d.mkdir(); tc = time.perf_counter()
    with open(d / 'raw.jsonl', 'w') as fh:
        for it in IT:
            text, batch = B[it['file']]; batch = batch.to(model.device)
            torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); sec = time.perf_counter() - s0
            raw = qp.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(condition='C0', file=it['file'], slurp_id=None, ref_sentence=it['sentence'], ref_intent=it['intent'],
                                     ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                                     new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1])), ensure_ascii=False) + '\n'); fh.flush()
    print('COND_DONE C0', round(time.perf_counter() - tc, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
