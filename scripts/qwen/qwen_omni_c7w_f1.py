"""One GPU job for docs/fasnet_single_talker_v1.md (F1, R1-same) and docs/c7_whisper_v1.md (C7-W), frozen before running.
  F1      : outputs/fasnet_single_talker_v1/streams/<key>.wav -> resample_poly 2/1 -> -30 dBFS -> frozen P2
  R1same  : outputs/fasnet_single_talker_v1/r1same/<key>.wav  -> same chain -> frozen P2
  W_<set> : Whisper large-v3 (revision 06f233fe..., local snapshot, fp16, en/transcribe, greedy, max_new_tokens 128)
            on C1 (124 originals, 16 kHz, no level change), C4direct (200; C4-direct chain), C5 (200; C5 chain)
  T_<set> : text-only frozen PROMPT_T2 on [whisper transcript.strip() + newline]
Qwen decoding identical to P2/T2. Order: F1, R1same, Whisper (all sets), T2 (all sets); each condition written and flushed.
--dry-run N: CPU only, no weights. Usage: python scripts/qwen/qwen_omni_c7w_f1.py --sep-root DIR --slurp-dir DIR --out DIR [--dry-run N]
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
from qwen_omni_single import REVISION, SR  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2  # noqa: E402
from qwen_omni_c3_p2 import LIST  # noqa: E402
from qwen_omni_c4 import level  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from slu_normalize import parse  # noqa: E402

WHISPER_REV = '06f233fe06e710322aca913c1bc4249a0d71fce1'
F1DIR = PROJ / 'outputs/fasnet_single_talker_v1'
SEL = PROJ / 'outputs/c5_v1/selection.jsonl'
REFS = PROJ / 'outputs/frontend_mix_v1_eval_v2/direct_refs'
DEPS = ['scripts/qwen/qwen_omni_c7w_f1.py', 'scripts/qwen/qwen_omni_c7w_f1.sbatch', 'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c4.py',
        'scripts/qwen/qwen_omni_c1_p2t2.py', 'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py',
        'scripts/sms_wsj/fasnet_single_talker.py', 'configs/mix_v1_devel100_v2.json', 'docs/fasnet_single_talker_v1.md', 'docs/c7_whisper_v1.md']


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={**{p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()},
                        'outputs/fasnet_single_talker_v1/f1_streams.jsonl': hashlib.sha256((F1DIR / 'f1_streams.jsonl').read_bytes()).hexdigest(),
                        'outputs/c5_v1/selection.jsonl': hashlib.sha256(SEL.read_bytes()).hexdigest()})


def chain8(x):
    y = level(resample_poly(x.astype(np.float64), 2, 1)[:2 * len(x)]); assert np.isfinite(y).all()
    return y.astype(np.float32)


def samples():
    out = []
    for g in sorted(json.load(open(LIST))['groups'], key=lambda g: g['group']):
        for sw in g['swaps']:
            t = g[sw['target']]
            out.append(dict(key=f"g{g['group']:03d}_s{sw['swap']}", group=g['group'], swap=sw['swap'], target=sw['target'], file=t['file'],
                            sentence=t['sentence'], intent=t['intent'], entities=t['entities']))
    return out


def build_audio(sep_root, slurp_dir):
    S = samples(); sel = {r['key']: r for r in map(json.loads, open(SEL))}
    A = dict(F1={}, R1same={}, C4direct={}, C5={}, C1={})
    for s in S:
        k = s['key']
        A['F1'][k] = chain8(sf.read(F1DIR / 'streams' / f'{k}.wav', dtype='float64')[0])
        A['R1same'][k] = chain8(sf.read(F1DIR / 'r1same' / f'{k}.wav', dtype='float64')[0])
        A['C4direct'][k] = chain8(np.load(REFS / f'{k}.npz')[s['target']])
        st = sel[k]['stream']; sign = -1.0 if sel[k]['flipped'][st] else 1.0
        A['C5'][k] = chain8(sign * sf.read(Path(sep_root) / 'fasnet_tac' / k / f'out{st}.wav', dtype='float64')[0])
    C1 = {}
    for s in S:
        if s['file'] not in C1:
            x, sr = sf.read(Path(slurp_dir) / s['file'], dtype='float32'); assert sr == SR and x.ndim == 1
            C1[s['file']] = dict(s, key=s['file'], wav=x)
    A['C1'] = {f: v['wav'] for f, v in C1.items()}
    return S, C1, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sep-root', type=Path, required=True); ap.add_argument('--slurp-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor, WhisperProcessor
    S, C1, A = build_audio(a.sep_root, a.slurp_dir); a.out.mkdir(parents=True, exist_ok=False)
    hub = Path(os.environ['HF_HOME']) / 'hub'
    qdir = hub / 'models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION; wdir = hub / 'models--openai--whisper-large-v3/snapshots' / WHISPER_REV
    assert qdir.is_dir() and (wdir / 'model.safetensors').exists()
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(qwen_dir=str(qdir), whisper_dir=str(wdir), whisper_revision=WHISPER_REV, prompt_P2=PROMPT_P2,
                                                       prompt_T2=PROMPT_T2, whisper_generate=dict(language='en', task='transcribe', num_beams=1,
                                                       do_sample=False, max_new_tokens=128), dry_run=a.dry_run,
                                                       specs=['docs/fasnet_single_talker_v1.md', 'docs/c7_whisper_v1.md'], **prov), indent=1) + '\n')
    print('AUDIO_BUILT', {k: len(v) for k, v in A.items()}, 'peak', {k: round(max(float(np.abs(w).max()) for w in v.values()), 3) for k, v in A.items()}, flush=True)
    sets = dict(C1=[dict(v, key=f) for f, v in C1.items()], C4direct=S, C5=S)
    if a.dry_run:
        qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True); wp = WhisperProcessor.from_pretrained(str(wdir), local_files_only=True)
        for it in S[:a.dry_run]:
            for name in ('F1', 'R1same'):
                text, batch = build(qp, PROMPT_P2, A[name][it['key']]); print('DRY', name, it['key'], list(batch['input_ids'].shape), flush=True)
            for name, items in sets.items():
                feat = wp(A[name][items[0]['key']], sampling_rate=SR, return_tensors='pt').input_features
                print('DRY whisper', name, list(feat.shape), flush=True)
            text, batch = build(qp, [it['sentence'] + '\n', PROMPT_T2]); print('DRY T2', list(batch['input_ids'].shape), flush=True)
        print('DRY_RUN_PASS', flush=True)
        return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, WhisperForConditionalGeneration
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa',
                                                                local_files_only=True)
    model.disable_talker(); qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True)
    wm = WhisperForConditionalGeneration.from_pretrained(str(wdir), torch_dtype=torch.float16, local_files_only=True).to('cuda').eval()
    wp = WhisperProcessor.from_pretrained(str(wdir), local_files_only=True)
    torch.cuda.synchronize(); print('LOAD', round(time.perf_counter() - t0, 1), 's', round(torch.cuda.max_memory_allocated() / 2**30, 2), 'GiB', flush=True)

    def qgen(text, batch):
        batch = batch.to(model.device)
        for k, v in batch.items():
            if torch.is_floating_point(v):
                batch[k] = v.to(model.dtype)
        with torch.inference_mode():
            ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
        return qp.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0], int(ids.shape[1] - batch['input_ids'].shape[1])

    def write(name, recs):
        d = a.out / name; d.mkdir()
        with open(d / 'raw.jsonl', 'w') as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    def rec(name, it, raw, sec, nt, **extra):
        return dict(condition=name, file=it['key'], group=it.get('group'), swap=it.get('swap'), target=it.get('target'), slurp_id=None,
                    ref_sentence=it['sentence'], ref_intent=it['intent'], ref_entities=it['entities'], raw_output=raw, parse_status=parse(raw)[1],
                    seconds=sec, new_tokens=nt, **extra)
    for name in ('F1', 'R1same'):
        t1 = time.perf_counter(); recs = []
        for it in S:
            text, batch = build(qp, PROMPT_P2, A[name][it['key']]); s0 = time.perf_counter(); raw, nt = qgen(text, batch)
            recs.append(rec(name, it, raw, time.perf_counter() - s0, nt))
        write(name, recs); print('COND_DONE', name, round(time.perf_counter() - t1, 1), flush=True)
    asr = {}
    for name, items in sets.items():
        t1 = time.perf_counter(); recs = []
        for it in items:
            feat = wp(A[name][it['key']], sampling_rate=SR, return_tensors='pt').input_features.to('cuda', torch.float16)
            s0 = time.perf_counter()
            with torch.inference_mode():
                ids = wm.generate(feat, language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128)
            txt = wp.batch_decode(ids, skip_special_tokens=True)[0]
            asr[(name, it['key'])] = txt; recs.append(rec(f'W_{name}', it, txt, time.perf_counter() - s0, int(ids.shape[1])))
        write(f'W_{name}', recs); print('COND_DONE', f'W_{name}', round(time.perf_counter() - t1, 1), flush=True)
    for name, items in sets.items():
        t1 = time.perf_counter(); recs = []
        for it in items:
            text, batch = build(qp, [asr[(name, it['key'])].strip() + '\n', PROMPT_T2]); s0 = time.perf_counter(); raw, nt = qgen(text, batch)
            recs.append(rec(f'T_{name}', it, raw, time.perf_counter() - s0, nt, asr_text=asr[(name, it['key'])]))
        write(f'T_{name}', recs); print('COND_DONE', f'T_{name}', round(time.perf_counter() - t1, 1), flush=True)
    print('PEAK_GPU_GIB', round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
