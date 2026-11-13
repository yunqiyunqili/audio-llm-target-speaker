"""C1 + C3 with the frozen P2 prompt (docs/c1_p2t2_results.md) on the mix_v1 devel list (configs/mix_v1_devel100_v2.json).
  C1: the 124 clean SLURP recordings used by the 100 groups (16 kHz originals), user content = [audio, PROMPT_P2].
  C3: the 200 mic0_16k mixtures (data/mix_v1/devel_seed2026, job 22013065), user content =
      [audio, C3_TEXT(theta = theta1 rounded half-up to integer degrees) + newline + PROMPT_P2] (docs/mix_spec_v1.md).
Decoding identical to P2 (greedy, max_new_tokens 256, batch 1, text only, use_audio_in_video=False). Frozen prompt code is
imported, not modified. OUT/config.json: git commit, dirty files, sha256 of dependencies, full prompts.
OUT/{C1,C3}/raw.jsonl in the rescore_slu_v2 format (C3 'file' = 'gNNN_sS', reference = the target speaker).
Gate / scoring rules: docs/c3_gate_v1.md (scripts/qwen/eval_c3_v1.py). --dry-run N: CPU only, no weights.
Usage: python scripts/qwen/qwen_omni_c3_p2.py --audio-dir SLURP_DIR --mix-dir MIX_DIR --out DIR [--dry-run N]
SLURP test (2026-11-12, docs/slurp_test_freeze_v1.md): env SLURP_EVAL_SPLIT=test selects the test mix list (split_cfg.LIST);
  --conds C3 --part K --nparts N runs C3 samples [K*n//N, (K+1)*n//N). Devel default (both conditions, all items) unchanged.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import REVISION, SR, SYSTEM  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2  # noqa: E402
from slu_normalize import parse  # noqa: E402

from split_cfg import LIST  # noqa: E402  (devel: configs/mix_v1_devel100_v2.json, unchanged)
C3_TEXT = ('The audio contains two overlapping speakers. Process only the speaker at azimuth {theta} degrees. '
           'Azimuth is measured from the array center, starting from the direction of microphone 0, counterclockwise '
           'positive, in the range -180 to 180 degrees.')
DEPS = ['scripts/qwen/qwen_omni_c3_p2.py', 'scripts/qwen/qwen_omni_c3_p2.sbatch', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py', 'scripts/qwen/slu_normalize.py',
        'configs/mix_v1_devel100_v2.json', 'configs/slurp_label_inventory_train.json', 'docs/c3_gate_v1.md']


def theta_int(x):
    return int(math.floor(x + 0.5))


def items(audio_dir, mix_dir):
    L = json.load(open(LIST))
    c1 = {}
    for g in L['groups']:
        for k in 'AB':
            s = g[k]; c1[s['file']] = dict(key=s['file'], path=str(Path(audio_dir) / s['file']), file=s['file'], slurp_id=None,
                                           sentence=s['sentence'], intent=s['intent'], entities=s['entities'])
    c3 = []
    for g in L['groups']:
        for sw in g['swaps']:
            t = g[sw['target']]; i = g['B' if sw['target'] == 'A' else 'A']
            key = f"g{g['group']:03d}_s{sw['swap']}"
            c3.append(dict(key=key, path=str(Path(mix_dir) / key / 'mic0_16k.wav'), file=key, group=g['group'], swap=sw['swap'],
                           target=sw['target'], target_file=t['file'], interferer_file=i['file'], theta1=g['rir']['theta1'],
                           theta_prompt=theta_int(g['rir']['theta1']), sentence=t['sentence'], intent=t['intent'],
                           entities=t['entities'], interferer_entities=i['entities']))
    return dict(C1=[c1[f] for f in sorted(c1)], C3=c3)


def prompt(cond, it):
    return PROMPT_P2 if cond == 'C1' else C3_TEXT.format(theta=it['theta_prompt']) + '\n' + PROMPT_P2


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()})


def build(processor, cond, it):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': prompt(cond, it)}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        assert len(text) == 1; text = text[0]
    wav, sr = sf.read(it['path'], dtype='float32')
    assert sr == SR and wav.ndim == 1, (it['path'], sr, wav.shape)
    return text, processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True), len(wav) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', type=Path, required=True); ap.add_argument('--mix-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--dry-run', type=int, default=0)
    ap.add_argument('--conds', default='C1,C3'); ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    IT = items(a.audio_dir, a.mix_dir)
    CONDS = tuple(a.conds.split(',')); assert CONDS in (('C1', 'C3'), ('C3',)) and (CONDS == ('C3',) or a.nparts == 1)
    IT = {c: v[a.part * len(v) // a.nparts:(a.part + 1) * len(v) // a.nparts] for c, v in IT.items() if c in CONDS}
    a.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    assert model_dir.is_dir(), model_dir
    prov = provenance(); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(
        model_dir=str(model_dir), model_commit=REVISION, list=str(LIST), mix_dir=str(a.mix_dir), n={k: len(v) for k, v in IT.items()},
        system=SYSTEM, prompt_C1=PROMPT_P2, c3_text_template=C3_TEXT, prompt_C3_template=C3_TEXT + '\n' + PROMPT_P2,
        theta_rounding='floor(theta1 + 0.5)', **(dict(conds=CONDS, part=a.part, nparts=a.nparts) if CONDS != ('C1', 'C3') else {}), decoding=dict(do_sample=False, max_new_tokens=a.max_new_tokens, batch=1,
        talker='disabled', use_audio_in_video=False), dry_run=a.dry_run, **prov), indent=1, ensure_ascii=False) + '\n')
    if a.dry_run:
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        for cond in CONDS:
            missing = [it['path'] for it in IT[cond] if not Path(it['path']).exists()]
            print('FILES', cond, len(IT[cond]), 'missing', len(missing), flush=True)
            for it in IT[cond][:a.dry_run]:
                text, batch, dur = build(processor, cond, it)
                fake = json.dumps(dict(transcript=it['sentence'], intent=it['intent'],
                                       entities=[dict(type=t, value=v) for t, v in it['entities']]))
                print('DRY', cond, it['key'], 'dur', round(dur, 2), 'chars', len(text), 'input_ids', list(batch['input_ids'].shape),
                      'parse', parse(fake)[1], flush=True)
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
        d = a.out / cond; d.mkdir()
        torch.cuda.reset_peak_memory_stats(); tot = 0.0; tc = time.perf_counter()
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in IT[cond]:
                text, batch, dur = build(processor, cond, it)
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
                rec = dict(condition=cond, file=it['key'], slurp_id=it.get('slurp_id'), ref_sentence=it['sentence'], ref_intent=it['intent'],
                           ref_entities=it['entities'], **{k: it[k] for k in ('group', 'swap', 'target', 'target_file', 'interferer_file',
                                                                              'theta1', 'theta_prompt', 'interferer_entities') if k in it},
                           audio=it['path'], audio_seconds=dur, prompt_text=text, raw_output=raw, parse_status=parse(raw)[1],
                           seconds=sec, input_tokens=int(batch['input_ids'].shape[1]), new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1]))
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
        summ = dict(condition=cond, n=len(IT[cond]), infer_seconds_total=tot, wall_seconds=time.perf_counter() - tc,
                    infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **load)
        (d / 'summary.json').write_text(json.dumps(summ, indent=2) + '\n')
        print('COND_DONE', json.dumps(summ), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
