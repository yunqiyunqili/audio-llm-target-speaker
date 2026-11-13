"""C4 v1 (docs/c4_v1.md, frozen): direct-reference-matched target stream of each front-end -> Qwen P2 (no direction text).
Target stream: ref_stream_direct in outputs/frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl (sha256 recorded).
Audio: separated_v1/<fe>/gNNN_sS/out{k}.wav (8 kHz) -> resample_poly(x, 2, 1)[:2T] -> scale to active-frame RMS -30 dBFS
(32 ms = 512-sample frames, >= peak - 40 dB); in memory only. Decoding identical to P2. Frozen prompt code imported.
--dry-run N: CPU only, no weights. Usage: python scripts/qwen/qwen_omni_c4.py --sep-root DIR --out DIR [--dry-run N]
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
from qwen_omni_single import REVISION, SR, SYSTEM  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2  # noqa: E402
from qwen_omni_c3_p2 import LIST  # noqa: E402
from slu_normalize import parse  # noqa: E402

FRONTENDS = ('blstm_mvdr', 'fasnet_tac')
SEL = PROJ / 'outputs/frontend_mix_v1_eval_v2/eval_v2_per_sample.jsonl'
LEVEL_DBFS, FRAME16, ACTIVE_DB = -30.0, 512, 40.0
DEPS = ['scripts/qwen/qwen_omni_c4.py', 'scripts/qwen/qwen_omni_c4.sbatch', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py', 'scripts/qwen/qwen_omni_c3_p2.py',
        'configs/mix_v1_devel100_v2.json', 'docs/c4_v1.md']


def level(x):
    n = len(x) // FRAME16
    p = (x[:n * FRAME16].reshape(n, FRAME16) ** 2).mean(1)
    P = p[p >= p.max() * 10 ** (-ACTIVE_DB / 10)].mean()
    return x * (10 ** (LEVEL_DBFS / 20) / np.sqrt(P))


def items(sep_root):
    L = {g['group']: g for g in json.load(open(LIST))['groups']}
    sel = {(r['frontend'], r['key']): r['ref_stream_direct'] for r in map(json.loads, open(SEL))}
    out = {fe: [] for fe in FRONTENDS}
    for fe in FRONTENDS:
        for g in sorted(L.values(), key=lambda g: g['group']):
            for sw in g['swaps']:
                key = f"g{g['group']:03d}_s{sw['swap']}"; t = g[sw['target']]; k = sel[(fe, key)]
                out[fe].append(dict(key=key, group=g['group'], swap=sw['swap'], target=sw['target'], stream=k,
                                    path=str(Path(sep_root) / fe / key / f'out{k}.wav'), T_8k=None,
                                    sentence=t['sentence'], intent=t['intent'], entities=t['entities']))
    return out


def audio(it):
    x, sr = sf.read(it['path'], dtype='float64'); assert sr == 8000 and x.ndim == 1
    y = level(resample_poly(x, 2, 1)[:2 * len(x)])
    assert np.isfinite(y).all()
    return y.astype(np.float32)


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={**{p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()},
                        str(SEL.relative_to(PROJ)): hashlib.sha256(SEL.read_bytes()).hexdigest()})


def build(processor, wav):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': PROMPT_P2}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        assert len(text) == 1; text = text[0]
    return text, processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sep-root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    IT = items(a.sep_root)
    a.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(
        model_dir=str(model_dir), model_commit=REVISION, spec='docs/c4_v1.md', n={k: len(v) for k, v in IT.items()}, system=SYSTEM,
        prompt=PROMPT_P2, level_dbfs=LEVEL_DBFS, frame_16k=FRAME16, upsample='resample_poly(x, 2, 1)[:2T]',
        decoding=dict(do_sample=False, max_new_tokens=a.max_new_tokens, batch=1, talker='disabled', use_audio_in_video=False),
        dry_run=a.dry_run, **prov), indent=1, ensure_ascii=False) + '\n')
    t0 = time.perf_counter(); AU = {fe: {it['key']: audio(it) for it in v} for fe, v in IT.items()}
    print('AUDIO_BUILT', {fe: len(v) for fe, v in AU.items()}, round(time.perf_counter() - t0, 1), 's',
          'peak_max', {fe: round(max(float(np.abs(w).max()) for w in v.values()), 3) for fe, v in AU.items()}, flush=True)
    if a.dry_run:
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        for fe in FRONTENDS:
            for it in IT[fe][:a.dry_run]:
                text, batch = build(processor, AU[fe][it['key']])
                print('DRY', fe, it['key'], 'stream', it['stream'], 'input_ids', list(batch['input_ids'].shape), flush=True)
                (a.out / f"dry_{fe}_{it['key']}.txt").write_text(text)
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
    for fe in FRONTENDS:
        d = a.out / fe; d.mkdir(); torch.cuda.reset_peak_memory_stats(); tot = 0.0
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in IT[fe]:
                text, batch = build(processor, AU[fe][it['key']])
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
                fh.write(json.dumps(dict(condition=f'C4_{fe}', file=it['key'], group=it['group'], swap=it['swap'], target=it['target'],
                                         stream=it['stream'], slurp_id=None, ref_sentence=it['sentence'], ref_intent=it['intent'],
                                         ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1],
                                         seconds=sec, new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1])), ensure_ascii=False) + '\n')
        (d / 'summary.json').write_text(json.dumps(dict(frontend=fe, n=len(IT[fe]), infer_seconds_total=tot,
                                                        infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **load), indent=2) + '\n')
        print('COND_DONE', fe, round(tot, 1), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
