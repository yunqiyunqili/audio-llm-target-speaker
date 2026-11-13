"""Step-1 audio LoRA on mix_v1 conditions that do not depend on SpatialNet (docs/lora_v2_runs.md §20-21; plan §6).
ONE condition per job. Inputs are built by the ORIGINAL frozen P2 scripts, imported unchanged:
  c4direct : qwen_omni_c4_direct.items() / audio() (direct reference, C4 chain) + qwen_omni_c4.build()   (baseline job 22015148)
  c3       : qwen_omni_c3_p2.items(...)['C3'] (mic0_16k mixture + direction text) + qwen_omni_c3_p2.build() (baseline job 22014089)
Base model + LoRA adapter loaded as in eval_audio_lora_ckpt.py; greedy, thinker_max_new_tokens 256, float inputs cast to bf16
(as in the original scripts). Output records keep the original fields (file = gNNN_sS, group, swap, target, ...).
--dry-run: CPU, processor only; also checks that the prompt text equals the baseline raw's prompt_text for the first items.
Usage: python scripts/lora/eval_audio_lora_mix.py --run-dir DIR --step S --cond {c4direct,c3} --out DIR [--dry-run]
SLURP test (2026-11-12, docs/slurp_test_freeze_v1.md): env SLURP_EVAL_SPLIT=test, --cond c3 --base RAW [RAW ...] (the test
  baseline C3 outputs) [--part K --nparts N]; test mix dir = split_cfg.MIX. Devel default unchanged.
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
from slu_normalize import parse  # noqa: E402
import qwen_omni_c3_p2 as C3  # noqa: E402
import qwen_omni_c4 as C4  # noqa: E402
import qwen_omni_c4_direct as C4D  # noqa: E402
import split_cfg as SC  # noqa: E402

TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'
SH = Path('<SHARED_ROOT>')
BASE = dict(c4direct=PROJ / 'outputs/qwen_omni_c4_direct/22015148/C4_direct/raw.jsonl', c3=PROJ / 'outputs/qwen_omni_c3_p2/22014089/C3/raw.jsonl')


def inputs(cond, qp):
    if cond == 'c4direct':
        IT = C4D.items()
        return [(it, lambda it=it: C4.build(qp, C4D.audio(it))) for it in IT]
    IT = C3.items(SH / 'data/slurp/audio/slurp_real', SC.MIX)['C3']  # devel: data/mix_v1/devel_seed2026 (unchanged)
    return [(it, lambda it=it: C3.build(qp, 'C3', it)[:2]) for it in IT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--cond', choices=('c4direct', 'c3'), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--base', type=Path, nargs='+'); ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    test = SC.SPLIT == 'test'
    assert (not test and a.base is None and a.nparts == 1) or (test and a.cond == 'c3' and a.base)
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    from transformers import Qwen2_5OmniProcessor
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    X = inputs(a.cond, qp)
    if not test:
        assert len(X) == 200
        base = {json.loads(l)['file']: json.loads(l) for l in open(BASE[a.cond])}; assert set(base) == {it['key'] for it, _ in X}
    else:
        assert len(X) == sum(1 for _ in open(SC.MANIFEST))
        base = {json.loads(l)['file']: json.loads(l) for p in a.base for l in open(p)}; assert set(base) == {it['key'] for it, _ in X}
        nX = len(X); X = X[a.part * nX // a.nparts:(a.part + 1) * nX // a.nparts]
    ck = a.run_dir / f'ckpt_step{a.step}' / 'adapter_model.safetensors'; assert ck.exists(), ck
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', 'scripts/lora', 'scripts/qwen').splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'), run_dir=str(a.run_dir), step=a.step,
                cond=a.cond, baseline=str(BASE[a.cond]) if not test else [str(p) for p in a.base], **(dict(split='test', part=a.part, nparts=a.nparts) if test else {}))
    print('ITEMS', len(X), a.cond, flush=True)
    if a.dry_run:
        same = 0
        for it, f in X[:5]:
            text, b = f(); same += text == base[it['key']]['prompt_text']
            print('DRY', it['key'], list(b['input_ids'].shape), flush=True)
        assert same == 5, same
        print('PROMPT_TEXT_EQUALS_BASELINE', same, '/ 5'); print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
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
    d = a.out / f'step{a.step}' / (a.cond + (f'_part{a.part}of{a.nparts}' if test else '')); d.mkdir(parents=True, exist_ok=False)
    (d / 'config.json').write_text(json.dumps(dict(lora_B_abs_sum=bsum, **prov), indent=1) + '\n')
    tc = time.perf_counter(); same = 0
    with open(d / 'raw.jsonl', 'w') as fh:
        for it, f in X:
            text, b = f(); same += text == base[it['key']]['prompt_text']; b = b.to(model.device)
            for k, v in b.items():
                if torch.is_floating_point(v):
                    b[k] = v.to(model.dtype)
            torch.cuda.synchronize(); s0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**b, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); raw = qp.batch_decode(ids[:, b['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            r = {k: v for k, v in base[it['key']].items() if k not in ('raw_output', 'parse_status', 'seconds', 'new_tokens', 'condition')}
            fh.write(json.dumps(dict(r, condition=f'{a.cond}_P2LoRA_step{a.step}', prompt_text=text, raw_output=raw, parse_status=parse(raw)[1],
                                     seconds=time.perf_counter() - s0, new_tokens=int(ids.shape[1] - b['input_ids'].shape[1])), ensure_ascii=False) + '\n'); fh.flush()
    print('PROMPT_TEXT_EQUALS_BASELINE', same, '/', len(X)); print('CKPT_DONE', a.step, round(time.perf_counter() - tc, 1), flush=True); print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
