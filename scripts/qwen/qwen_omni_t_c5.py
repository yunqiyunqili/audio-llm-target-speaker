"""Rerun of T_C5 only (docs/c7_whisper_v1.md, rules unchanged): reads the existing Whisper transcripts of job 22016221
(W_C5/raw.jsonl), then frozen PROMPT_T2 on [transcript.strip() + newline] exactly as qwen_omni_c7w_f1.py; no Whisper rerun.
Records commit, node, GPU model and per-item seconds. --dry-run N: CPU only.
Usage: python scripts/qwen/qwen_omni_t_c5.py --w-c5 PATH --out DIR [--dry-run N]
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

DEPS = ['scripts/qwen/qwen_omni_t_c5.py', 'scripts/qwen/qwen_omni_t_c5.sbatch', 'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py', 'docs/c7_whisper_v1.md']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w-c5', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    W = [json.loads(l) for l in open(a.w_c5)]; assert len(W) == 200
    a.out.mkdir(parents=True, exist_ok=False)
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={**{p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS}, str(a.w_c5): hashlib.sha256(a.w_c5.read_bytes()).hexdigest()})
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    if a.dry_run:
        for w in W[:a.dry_run]:
            text, batch = build(qp, [w['raw_output'].strip() + '\n', PROMPT_T2]); print('DRY', w['file'], list(batch['input_ids'].shape), flush=True)
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True)
        return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration
    prov['gpu'] = torch.cuda.get_device_name(0)
    print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(prompt_T2=PROMPT_T2, source=str(a.w_c5), **prov), indent=1) + '\n')
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa',
                                                                local_files_only=True)
    model.disable_talker(); torch.cuda.synchronize(); print('LOAD', round(time.perf_counter() - t0, 1), flush=True)
    d = a.out / 'T_C5'; d.mkdir(); tot = 0.0
    with open(d / 'raw.jsonl', 'w') as fh:
        for w in W:
            text, batch = build(qp, [w['raw_output'].strip() + '\n', PROMPT_T2]); batch = batch.to(model.device)
            torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); sec = time.perf_counter() - s0; tot += sec
            raw = qp.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(condition='T_C5', file=w['file'], group=w['group'], swap=w['swap'], target=w['target'], slurp_id=None,
                                     ref_sentence=w['ref_sentence'], ref_intent=w['ref_intent'], ref_entities=w['ref_entities'], raw_output=raw,
                                     parse_status=parse(raw)[1], seconds=sec, new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1]),
                                     asr_text=w['raw_output']), ensure_ascii=False) + '\n'); fh.flush()
    (d / 'summary.json').write_text(json.dumps(dict(n=len(W), infer_seconds_total=tot, mean_seconds=tot / len(W), gpu=prov['gpu'], node=prov['node'],
                                                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30), indent=2) + '\n')
    print('ALL_DONE', round(tot, 1), 'mean', round(tot / len(W), 3), flush=True)


if __name__ == '__main__':
    main()
