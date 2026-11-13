"""C3 factor diagnostic v1 (docs/c3_factor_diag_v1.md, frozen before submission). Audio generated IN MEMORY only.
Single-speaker (124 recordings of configs/mix_v1_devel100_v2.json), P2: C1b, C2 (early RIR, mic 0), R1 (full RIR, mic 0).
  RIR = cv_dev93 example of the recording's first group; position from swap 1 (A -> h_0, B -> h_1).
Mixtures (the 40 samples of job 22014131, scripts/qwen/qwen_omni_c3_diag.py: items): X1 dry 2-speaker 16 kHz 0 dB (P2),
  X2 = X1 through 16k->8k->16k (P2), X3 = existing mic0_16k with P0.
Frozen code is imported (mix_slurp_v1: rir_start / extract_piece / active_power / EARLY / LEVEL_DBFS; qwen_omni_single.c1b;
prompts). Decoding identical to P2. --dry-run N: CPU only, no weights: builds ALL audio, runs the processor on N items.
Usage: python scripts/qwen/qwen_omni_c3_factor.py --audio-dir SLURP_DIR --mix-dir MIX_DIR --out DIR [--dry-run N]
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
from scipy.signal import fftconvolve, resample_poly

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(PROJ / 'scripts/sms_wsj'))
from qwen_omni_single import REVISION, SR, SYSTEM, c1b  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2  # noqa: E402
from qwen_omni_c1_semantic import PROMPT as PROMPT_P0  # noqa: E402
from qwen_omni_c3_diag import items as diag_items  # noqa: E402
from qwen_omni_c3_p2 import LIST  # noqa: E402
from mix_slurp_v1 import EARLY, LEVEL_DBFS, active_power, extract_piece, rir_start  # noqa: E402
from slu_normalize import parse  # noqa: E402

SINGLE, MIXED = ('C1b', 'C2', 'R1'), ('X1', 'X2', 'X3')
DEPS = ['scripts/qwen/qwen_omni_c3_factor.py', 'scripts/qwen/qwen_omni_c3_factor.sbatch', 'scripts/qwen/qwen_omni_c3_diag.py',
        'scripts/qwen/qwen_omni_c3_p2.py', 'scripts/qwen/qwen_omni_c1_p2t2.py', 'scripts/qwen/qwen_omni_c1_semantic.py',
        'scripts/qwen/qwen_omni_single.py', 'scripts/sms_wsj/mix_slurp_v1.py', 'configs/mix_v1_devel100_v2.json',
        'docs/c3_factor_diag_v1.md']


def gain_to_level(x):
    return 10 ** (LEVEL_DBFS / 20) / np.sqrt(active_power(x)[0])


def single_items(audio_dir):
    L = json.load(open(LIST)); first = {}
    for g in sorted(L['groups'], key=lambda g: g['group']):
        for k, pos in (('A', 0), ('B', 1)):
            s = g[k]
            if s['file'] not in first:
                first[s['file']] = dict(key=s['file'], path=str(Path(audio_dir) / s['file']), sentence=s['sentence'], intent=s['intent'],
                                        entities=s['entities'], rir_dir=g['rir']['rir_dir'], rir_pos=pos, rir_group=g['group'])
    return [first[f] for f in sorted(first)]


def reverb(x16, rir_dir, pos):
    s8 = resample_poly(x16.astype(np.float64), 1, 2); T = len(s8)
    h, sr = sf.read(Path(rir_dir) / f'h_{pos}.wav', dtype='float64', always_2d=True); h = h.T; assert sr == 8000
    rs = rir_start(h); he = h.copy(); he[:, rs + EARLY:] = 0
    early = extract_piece(fftconvolve(s8, he[0]), -rs, T); full = extract_piece(fftconvolve(s8, h[0]), -rs, T)
    g = gain_to_level(early)
    up = lambda y: resample_poly(y * g, 2, 1)[:2 * T].astype(np.float32)  # noqa: E731
    return up(early), up(full), dict(rir_start=rs, gain=g)


def audio_single(it):
    x, sr = sf.read(it['path'], dtype='float32'); assert sr == SR and x.ndim == 1
    e, f, info = reverb(x, it['rir_dir'], it['rir_pos'])
    return dict(C1b=c1b(x), C2=e, R1=f), info


def audio_mixed(it, L, audio_dir):
    g = L['groups'][it['group']]
    xs = {}
    for k in 'AB':
        x, sr = sf.read(Path(audio_dir) / g[k]['file'], dtype='float64'); assert sr == SR and len(x) == g[k]['n_16k']
        xs[k] = (x * gain_to_level(x), 2 * g[k]['offset_8k'])
    T = max(off + len(x) for x, off in xs.values())
    x1 = np.zeros(T)
    for x, off in xs.values():
        x1[off:off + len(x)] += x
    x2 = resample_poly(resample_poly(x1, 1, 2), 2, 1)[:T]
    x3, sr = sf.read(it['path'], dtype='float32'); assert sr == SR
    return dict(X1=x1.astype(np.float32), X2=x2.astype(np.float32), X3=x3), dict(T_16k=T, peak_X1=float(np.abs(x1).max()))


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()})


def build(processor, cond, wav):
    text_prompt = PROMPT_P0 if cond == 'X3' else PROMPT_P2
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': text_prompt}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        assert len(text) == 1; text = text[0]
    return text, processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', type=Path, required=True); ap.add_argument('--mix-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    L = json.load(open(LIST))
    S = single_items(a.audio_dir); gids, M = diag_items(a.mix_dir)
    a.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(
        model_dir=str(model_dir), model_commit=REVISION, spec='docs/c3_factor_diag_v1.md', n_single=len(S), n_mixed=len(M), groups=gids,
        system=SYSTEM, prompt_P2=PROMPT_P2, prompt_P0_for_X3=PROMPT_P0, level_dbfs=LEVEL_DBFS,
        decoding=dict(do_sample=False, max_new_tokens=a.max_new_tokens, batch=1, talker='disabled', use_audio_in_video=False),
        dry_run=a.dry_run, **prov), indent=1, ensure_ascii=False) + '\n')
    t0 = time.perf_counter()
    AS, AM, stats = {}, {}, []
    for it in S:
        AS[it['key']], info = audio_single(it); stats.append(dict(key=it['key'], kind='single', **info,
                                                                   **{f'peak_{c}': float(np.abs(v).max()) for c, v in AS[it['key']].items()}))
    for it in M:
        AM[it['key']], info = audio_mixed(it, L, a.audio_dir); stats.append(dict(key=it['key'], kind='mixed', **info))
    bad = [s['key'] for s in stats for k, v in s.items() if k.startswith('peak') and v >= 1]
    nonfinite = [k for d in (AS, AM) for k, v in d.items() for w in v.values() if not np.isfinite(w).all()]
    with open(a.out / 'audio_stats.jsonl', 'w') as fh:
        for s in stats:
            fh.write(json.dumps(s) + '\n')
    print('AUDIO_BUILT', len(AS), len(AM), 'seconds', round(time.perf_counter() - t0, 1), 'clipped', len(bad), 'nonfinite', len(nonfinite), flush=True)
    assert not bad and not nonfinite
    if a.dry_run:
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        for cond in SINGLE:
            for it in S[:a.dry_run]:
                text, batch = build(processor, cond, AS[it['key']][cond])
                print('DRY', cond, it['key'], 'input_ids', list(batch['input_ids'].shape), flush=True)
        for cond in MIXED:
            for it in M[:a.dry_run]:
                text, batch = build(processor, cond, AM[it['key']][cond])
                print('DRY', cond, it['key'], 'input_ids', list(batch['input_ids'].shape), flush=True)
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
    for cond in SINGLE + MIXED:
        its, A = (S, AS) if cond in SINGLE else (M, AM)
        d = a.out / cond; d.mkdir(); torch.cuda.reset_peak_memory_stats(); tot = 0.0
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in its:
                text, batch = build(processor, cond, A[it['key']][cond])
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
                fh.write(json.dumps(dict(condition=cond, file=it['key'], slurp_id=None, ref_sentence=it['sentence'], ref_intent=it['intent'],
                                         ref_entities=it['entities'], **{k: it[k] for k in ('group', 'swap', 'target') if k in it},
                                         prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                                         new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1])), ensure_ascii=False) + '\n')
        (d / 'summary.json').write_text(json.dumps(dict(condition=cond, n=len(its), infer_seconds_total=tot,
                                                        infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **load), indent=2) + '\n')
        print('COND_DONE', cond, round(tot, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
