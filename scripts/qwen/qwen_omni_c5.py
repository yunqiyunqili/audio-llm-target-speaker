"""C5 v1 (docs/c5_v1.md, frozen): FaSNet-TAC stream chosen by outputs/c5_v1/selection.jsonl (v3 polarity + direction
matching, oracle query azimuth) -> polarity applied -> same chain as C4 (resample_poly 2/1, -30 dBFS active level) ->
frozen P2, no direction text. Selection file sha256 recorded. --dry-run N: CPU only.
Usage: python scripts/qwen/qwen_omni_c5.py --sep-root DIR --out DIR [--dry-run N]
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
from qwen_omni_single import REVISION  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2  # noqa: E402
from qwen_omni_c3_p2 import LIST  # noqa: E402
from qwen_omni_c4 import LEVEL_DBFS, build, level  # noqa: E402
from slu_normalize import parse  # noqa: E402

SEL = PROJ / 'outputs/c5_v1/selection.jsonl'
DEPS = ['scripts/qwen/qwen_omni_c5.py', 'scripts/qwen/qwen_omni_c5.sbatch', 'scripts/qwen/qwen_omni_c4.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py', 'scripts/sms_wsj/c5_select.py',
        'configs/mix_v1_devel100_v2.json', 'docs/c5_v1.md']


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
            key = f"g{g['group']:03d}_s{sw['swap']}"; t = g[sw['target']]; s = sel[key]; k = s['stream']
            out.append(dict(key=key, group=g['group'], swap=sw['swap'], target=sw['target'], stream=k, sign=-1.0 if s['flipped'][k] else 1.0,
                            path=str(Path(sep_root) / 'fasnet_tac' / key / f'out{k}.wav'), sentence=t['sentence'], intent=t['intent'], entities=t['entities']))
    return out


def audio(it):
    x, sr = sf.read(it['path'], dtype='float64'); assert sr == 8000 and x.ndim == 1
    y = level(resample_poly(it['sign'] * x, 2, 1)[:2 * len(x)]); assert np.isfinite(y).all()
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sep-root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    IT = items(a.sep_root); a.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(model_dir=str(model_dir), model_commit=REVISION, spec='docs/c5_v1.md', n=len(IT),
                                                       prompt=PROMPT_P2, level_dbfs=LEVEL_DBFS, oracle='query azimuth theta1 (ground truth)',
                                                       dry_run=a.dry_run, **prov), indent=1, ensure_ascii=False) + '\n')
    AU = {it['key']: audio(it) for it in IT}
    print('AUDIO_BUILT', len(AU), 'peak_max', round(max(float(np.abs(w).max()) for w in AU.values()), 3), flush=True)
    if a.dry_run:
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        for it in IT[:a.dry_run]:
            text, batch = build(processor, AU[it['key']])
            print('DRY', it['key'], 'stream', it['stream'], 'sign', it['sign'], 'input_ids', list(batch['input_ids'].shape), flush=True)
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
    d = a.out / 'C5_fasnet_tac'; d.mkdir(); tot = 0.0
    with open(d / 'raw.jsonl', 'w') as fh:
        for it in IT:
            text, batch = build(processor, AU[it['key']])
            batch = batch.to(model.device)
            for k, v in batch.items():
                if torch.is_floating_point(v):
                    batch[k] = v.to(model.dtype)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=a.max_new_tokens, do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); sec = time.perf_counter() - t1; tot += sec
            raw = processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            fh.write(json.dumps(dict(condition='C5_fasnet_tac', file=it['key'], group=it['group'], swap=it['swap'], target=it['target'],
                                     stream=it['stream'], sign=it['sign'], slurp_id=None, ref_sentence=it['sentence'], ref_intent=it['intent'],
                                     ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                                     new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1])), ensure_ascii=False) + '\n')
    (d / 'summary.json').write_text(json.dumps(dict(n=len(IT), infer_seconds_total=tot, infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                                                    **load), indent=2) + '\n')
    print('ALL_DONE', round(tot, 1), flush=True)


if __name__ == '__main__':
    main()
