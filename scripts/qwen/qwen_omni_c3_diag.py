"""C3 diagnostic run on 20 groups (40 mic0_16k mixtures) drawn with random.Random(2026).sample(range(100), 20).
Same audio for all three prompts; same decoding as P2 / C3 (greedy, max_new_tokens 256, batch 1, text only):
  D1: user text = "Transcribe all speech in this audio."  (no JSON, no labels, no direction)
  D2: frozen P2 prompt, no direction text
  D3: frozen C3 prompt (direction text + P2), identical to job 22014089 -> raw outputs compared afterwards
Frozen prompt code is imported, not modified. OUT/config.json: commit, dirty files, sha256, full prompts, chosen groups.
--dry-run N: CPU only, no weights. Usage: python scripts/qwen/qwen_omni_c3_diag.py --mix-dir DIR --out DIR [--dry-run N]
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import REVISION, SR, SYSTEM  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2  # noqa: E402
from qwen_omni_c3_p2 import C3_TEXT, LIST, theta_int  # noqa: E402
from slu_normalize import parse  # noqa: E402

D1_TEXT = 'Transcribe all speech in this audio.'
CONDS = ('D1', 'D2', 'D3')
DEPS = ['scripts/qwen/qwen_omni_c3_diag.py', 'scripts/qwen/qwen_omni_c3_diag.sbatch', 'scripts/qwen/qwen_omni_c3_p2.py',
        'scripts/qwen/qwen_omni_c1_p2t2.py', 'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py',
        'configs/mix_v1_devel100_v2.json']


def items(mix_dir):
    L = json.load(open(LIST))
    gids = sorted(random.Random(2026).sample(range(len(L['groups'])), 20))
    out = []
    for gi in gids:
        g = L['groups'][gi]
        for sw in g['swaps']:
            key = f"g{g['group']:03d}_s{sw['swap']}"; t = g[sw['target']]
            out.append(dict(key=key, group=g['group'], swap=sw['swap'], target=sw['target'], path=str(Path(mix_dir) / key / 'mic0_16k.wav'),
                            theta_prompt=theta_int(g['rir']['theta1']), sentence=t['sentence'], intent=t['intent'], entities=t['entities']))
    return gids, out


def prompt(cond, it):
    return D1_TEXT if cond == 'D1' else PROMPT_P2 if cond == 'D2' else C3_TEXT.format(theta=it['theta_prompt']) + '\n' + PROMPT_P2


def build(processor, cond, it):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': prompt(cond, it)}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        assert len(text) == 1; text = text[0]
    wav, sr = sf.read(it['path'], dtype='float32')
    assert sr == SR and wav.ndim == 1
    return text, processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix-dir', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    gids, IT = items(a.mix_dir)
    a.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(
        model_dir=str(model_dir), model_commit=REVISION, groups=gids, n=len(IT), system=SYSTEM, prompt_D1=D1_TEXT, prompt_D2=PROMPT_P2,
        prompt_D3_template=C3_TEXT + '\n' + PROMPT_P2, decoding=dict(do_sample=False, max_new_tokens=a.max_new_tokens, batch=1,
        talker='disabled', use_audio_in_video=False), dry_run=a.dry_run, **prov), indent=1, ensure_ascii=False) + '\n')
    print('GROUPS', gids, flush=True)
    if a.dry_run:
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        print('FILES', len(IT), 'missing', sum(not Path(it['path']).exists() for it in IT), flush=True)
        for it in IT[:a.dry_run]:
            for cond in CONDS:
                text, batch = build(processor, cond, it)
                print('DRY', cond, it['key'], 'chars', len(text), 'input_ids', list(batch['input_ids'].shape), flush=True)
                (a.out / f"dry_{cond}_{it['key']}.txt").write_text(text)
        print('DRY_RUN_PASS', flush=True)
        return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
    torch.cuda.synchronize(); load = dict(load_seconds=time.perf_counter() - t0, load_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
    print('LOAD', json.dumps(load), flush=True)
    for cond in CONDS:
        d = a.out / cond; d.mkdir(); torch.cuda.reset_peak_memory_stats(); tot = 0.0
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in IT:
                text, batch = build(processor, cond, it)
                batch = batch.to(model.device)
                for k, v in batch.items():
                    if torch.is_floating_point(v):
                        batch[k] = v.to(model.dtype)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=a.max_new_tokens,
                                         do_sample=False, use_audio_in_video=False)
                torch.cuda.synchronize(); sec = time.perf_counter() - t1; tot += sec
                raw = processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                fh.write(json.dumps(dict(condition=cond, file=it['key'], group=it['group'], swap=it['swap'], target=it['target'],
                                         slurp_id=None, ref_sentence=it['sentence'], ref_intent=it['intent'], ref_entities=it['entities'],
                                         prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                                         input_tokens=int(batch['input_ids'].shape[1]),
                                         new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1])), ensure_ascii=False) + '\n')
        (d / 'summary.json').write_text(json.dumps(dict(condition=cond, n=len(IT), infer_seconds_total=tot,
                                                        infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **load), indent=2) + '\n')
        print('COND_DONE', cond, round(tot, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
