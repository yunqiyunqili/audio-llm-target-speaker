"""DIAGNOSTIC (docs/lora_v2_runs.md §13-3; plan §6 item 5): step-0 text LoRA checkpoint on Whisper transcripts of the 428 devel
clean recordings ("text LoRA does not degrade C1"). Not a step-2 result. Same model / LoRA loading / decoding as
eval_text_lora_ckpt.py; input = [whisper transcript.strip() + newline, frozen T2] exactly as the untuned T path of
baseline_audio.py (whisper_t2 devel_clean), which is the comparison. One generation per recording (transcripts differ).
--part K --nparts N over recordings sorted by file. --dry-run: CPU, tokenizer only.
Usage: python scripts/lora/eval_text_lora_asr.py --run-dir DIR --step S --asr W_RAW [W_RAW ...] --out DIR [--part K --nparts N] [--dry-run]
"""
import argparse
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--asr', type=Path, nargs='+', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    R = {}
    for p in a.asr:
        for l in open(p):
            r = json.loads(l); assert r['condition'] == 'W' and r['audio'] == 'clean' and r['file'] not in R; R[r['file']] = r
    assert len(R) == 428, len(R)
    rows = [R[f] for f in sorted(R)]; n = len(rows); rows = rows[a.part * n // a.nparts:(a.part + 1) * n // a.nparts]
    from transformers import Qwen2_5OmniProcessor
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    B = {r['file']: build(qp, [r['raw_output'].strip() + '\n', PROMPT_T2]) for r in rows}
    ck = a.run_dir / f'ckpt_step{a.step}' / 'adapter_model.safetensors'; assert ck.exists(), ck
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(diagnostic=True, git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', 'scripts/lora', 'scripts/qwen').splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'), run_dir=str(a.run_dir), step=a.step,
                asr=[str(p) for p in a.asr], part=a.part, nparts=a.nparts, n_part=len(rows))
    print('ITEMS', len(rows), 'of', n, flush=True)
    if a.dry_run:
        for r in rows[:2]:
            print('DRY', r['file'], list(B[r['file']][1]['input_ids'].shape), repr(r['raw_output'][:80]), flush=True)
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
    res = set_peft_model_state_dict(pm, load_file(str(ck)))
    assert not [k for k in getattr(res, 'unexpected_keys', []) if 'lora' in k], res
    bsum = float(sum(q.detach().double().abs().sum() for nm, q in pm.named_parameters() if 'lora_B' in nm)); assert bsum > 0
    print('ADAPTER', a.step, 'lora_B_abs_sum', round(bsum, 4), flush=True)
    d = a.out / 'T0LoRA_W'; d.mkdir(parents=True, exist_ok=False)
    (d.parent / 'config.json').write_text(json.dumps(dict(prompt_T2=PROMPT_T2, **prov), indent=1, ensure_ascii=False) + '\n')
    tc = time.perf_counter()
    with open(d / 'raw.jsonl', 'w') as fh:
        for r in rows:
            text, b = B[r['file']]; b = b.to(model.device); torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**b, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); raw = qp.batch_decode(ids[:, b['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(condition=f'T0LoRA_step{a.step}_W', diagnostic=True, file=r['file'], audio='clean', slurp_id=r['slurp_id'], ref_sentence=r['ref_sentence'],
                                     ref_intent=r['ref_intent'], ref_entities=r['ref_entities'], asr_text=r['raw_output'], prompt_text=text, raw_output=raw,
                                     parse_status=parse(raw)[1], seconds=time.perf_counter() - s0, new_tokens=int(ids.shape[1] - b['input_ids'].shape[1])),
                                ensure_ascii=False) + '\n'); fh.flush()
    print('CKPT_DONE', a.step, round(time.perf_counter() - tc, 1), flush=True); print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
