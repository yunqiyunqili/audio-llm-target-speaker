"""One GPU job for docs/c6_delta_v1.md and docs/c7_planA_v1.md (both frozen before running).
  OTHER : the FaSNet-TAC stream NOT chosen by C5 (1 - selection.stream), v3 polarity applied, C4/C5 chain, frozen P2.
  C7ASR : C5 target-stream audio (identical to C5 input) + qwen_omni_single.PROMPT (transcription only).
  C7T2  : text-only; [ASR transcript.strip() + newline, frozen PROMPT_T2].
Decoding identical to P2 (greedy, max_new_tokens 256, batch 1, text only). Frozen prompt code imported.
--dry-run N: CPU only, no weights (C7T2 dry-run uses the reference sentence as a stand-in transcript).
Usage: python scripts/qwen/qwen_omni_c6c7.py --sep-root DIR --out DIR [--dry-run N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import PROMPT as PROMPT_ASR, REVISION, SR, SYSTEM  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2  # noqa: E402
from qwen_omni_c3_p2 import LIST  # noqa: E402
from qwen_omni_c4 import level  # noqa: E402
from slu_normalize import parse  # noqa: E402

SEL = PROJ / 'outputs/c5_v1/selection.jsonl'
DEPS = ['scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c6c7.sbatch', 'scripts/qwen/qwen_omni_c4.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py', 'configs/mix_v1_devel100_v2.json',
        'docs/c6_delta_v1.md', 'docs/c7_planA_v1.md']


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={**{p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()},
                        str(SEL.relative_to(PROJ)): hashlib.sha256(SEL.read_bytes()).hexdigest()})


def items(sep_root):
    sel = {r['key']: r for r in map(json.loads, open(SEL))}
    out = []
    for g in sorted(json.load(open(LIST))['groups'], key=lambda g: g['group']):
        for sw in g['swaps']:
            key = f"g{g['group']:03d}_s{sw['swap']}"; t = g[sw['target']]; s = sel[key]
            base = dict(key=key, group=g['group'], swap=sw['swap'], target=sw['target'], sentence=t['sentence'], intent=t['intent'], entities=t['entities'])
            for role, k in (('other', 1 - s['stream']), ('c5', s['stream'])):
                base[f'{role}_stream'] = k; base[f'{role}_sign'] = -1.0 if s['flipped'][k] else 1.0
                base[f'{role}_path'] = str(Path(sep_root) / 'fasnet_tac' / key / f'out{k}.wav')
            out.append(base)
    return out


def audio(path, sign):
    x, sr = sf.read(path, dtype='float64'); assert sr == 8000 and x.ndim == 1
    y = level(resample_poly(sign * x, 2, 1)[:2 * len(x)]); assert np.isfinite(y).all()
    return y.astype(np.float32)


def build(processor, user_text, wav=None):
    content = ([{'type': 'audio', 'audio': 'placeholder'}] if wav is not None else []) + \
              ([{'type': 'text', 'text': t} for t in user_text] if isinstance(user_text, list) else [{'type': 'text', 'text': user_text}])
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': content}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        assert len(text) == 1; text = text[0]
    if wav is None:
        return text, processor(text=text, return_tensors='pt', padding=True)
    return text, processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sep-root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    IT = items(a.sep_root); a.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(model_dir=str(model_dir), model_commit=REVISION, specs=['docs/c6_delta_v1.md', 'docs/c7_planA_v1.md'],
                                                       n=len(IT), prompt_P2=PROMPT_P2, prompt_ASR=PROMPT_ASR, prompt_T2=PROMPT_T2, dry_run=a.dry_run,
                                                       oracle='C7 input = C5 target stream (oracle query azimuth)', **prov), indent=1, ensure_ascii=False) + '\n')
    AO = {it['key']: audio(it['other_path'], it['other_sign']) for it in IT}
    AC = {it['key']: audio(it['c5_path'], it['c5_sign']) for it in IT}
    print('AUDIO_BUILT', len(AO), len(AC), flush=True)
    if a.dry_run:
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        for it in IT[:a.dry_run]:
            for name, (ut, w) in dict(OTHER=(PROMPT_P2, AO[it['key']]), C7ASR=(PROMPT_ASR, AC[it['key']]),
                                      C7T2=([it['sentence'] + '\n', PROMPT_T2], None)).items():
                text, batch = build(processor, ut, w)
                print('DRY', name, it['key'], 'input_ids', list(batch['input_ids'].shape), flush=True)
                (a.out / f"dry_{name}_{it['key']}.txt").write_text(text)
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

    def gen(text, batch):
        batch = batch.to(model.device)
        for k, v in batch.items():
            if torch.is_floating_point(v):
                batch[k] = v.to(model.dtype)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=a.max_new_tokens, do_sample=False, use_audio_in_video=False)
        torch.cuda.synchronize()
        return processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0], time.perf_counter() - t1, \
            int(ids.shape[1] - batch['input_ids'].shape[1])
    asr = {}
    for name in ('OTHER', 'C7ASR', 'C7T2'):
        d = a.out / name; d.mkdir(); tot = 0.0; torch.cuda.reset_peak_memory_stats()
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in IT:
                if name == 'OTHER':
                    text, batch = build(processor, PROMPT_P2, AO[it['key']]); extra = dict(stream=it['other_stream'], sign=it['other_sign'])
                elif name == 'C7ASR':
                    text, batch = build(processor, PROMPT_ASR, AC[it['key']]); extra = dict(stream=it['c5_stream'], sign=it['c5_sign'])
                else:
                    text, batch = build(processor, [asr[it['key']].strip() + '\n', PROMPT_T2]); extra = dict(asr_text=asr[it['key']])
                raw, sec, nt = gen(text, batch); tot += sec
                if name == 'C7ASR':
                    asr[it['key']] = raw
                fh.write(json.dumps(dict(condition=name, file=it['key'], group=it['group'], swap=it['swap'], target=it['target'], slurp_id=None,
                                         ref_sentence=it['sentence'], ref_intent=it['intent'], ref_entities=it['entities'], prompt_text=text,
                                         raw_output=raw, parse_status=parse(raw)[1], seconds=sec, new_tokens=nt, **extra), ensure_ascii=False) + '\n')
        (d / 'summary.json').write_text(json.dumps(dict(condition=name, n=len(IT), infer_seconds_total=tot,
                                                        infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **load), indent=2) + '\n')
        print('COND_DONE', name, round(tot, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
