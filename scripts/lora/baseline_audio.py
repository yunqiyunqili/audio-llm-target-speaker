"""Audio baselines / transcription for LoRA steps 1-2 (docs/lora_slu_plan_v2.md §5, §8, §11-4). One script, sharded.
Modes (decoding identical to the frozen C1 / C7-W paths: qwen_omni_c7w_f1.py):
  p2         : audio -> frozen P2 (Qwen2.5-Omni-7B ae9e1690, bf16, talker off, greedy, thinker_max_new_tokens 256) -> P2/raw.jsonl
  whisper_t2 : audio -> Whisper large-v3 06f233fe (fp16, en/transcribe, greedy, max_new_tokens 128) -> W/raw.jsonl;
               then [transcript.strip() + newline] -> frozen T2 -> T/raw.jsonl
  whisper    : Whisper only -> W/raw.jsonl (train transcription for step 2)
Item sets (all sorted by file):
  devel_clean : 428 split_v2 devel recordings, SLURP 16 kHz as is (no level change; = C1 / W_C1)
  devel_rev   : same 428, single-talker full reverb (cv_dev93 RIR; scripts/lora/reverb_single.py output)
  train_all   : 3,513 train clean + 3,513 train reverb (train_si284 RIR) = 7,026 items, clean first then reverb
  test_clean / test_rev : SLURP test (docs/slurp_test_freeze_v1.md): 1,183 split_v2 test recordings, as devel_clean / devel_rev
                (test_rev = test_eval92 RIR, reverb_single.py --split test). Added 2026-11-12; devel/train sets unchanged.
--part K --nparts N: items [K*n//N, (K+1)*n//N). Output format = C0-428 raw.jsonl (+ audio, asr_text). Records commit, dirty deps,
sha256, node, GPU, per-item seconds. --dry-run M: CPU, processors only, no weights; loads every audio item of the part (reverb
items that are not generated yet are computed in memory with reverb_single.reverb) and builds the first M model inputs.
Usage: python scripts/lora/baseline_audio.py --mode MODE --items SET --out DIR [--part K --nparts N] [--dry-run M]
"""
import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/lora'))
from qwen_omni_single import REVISION, SR  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_P2, PROMPT_T2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from slu_normalize import parse  # noqa: E402
import reverb_single as RV  # noqa: E402

WHISPER_REV = '06f233fe06e710322aca913c1bc4249a0d71fce1'
DATA = Path('<SHARED_ROOT>/data/lora_v1')
REVDIR = dict(devel=DATA / 'devel_rev_cv_dev93', train=DATA / 'train_rev_si284', test=DATA / 'test_rev_eval92')
DEPS = ['scripts/lora/baseline_audio.py', 'scripts/lora/baseline_audio.sbatch', 'scripts/lora/reverb_single.py', 'scripts/qwen/qwen_omni_c6c7.py',
        'scripts/qwen/qwen_omni_c1_p2t2.py', 'scripts/qwen/qwen_omni_single.py', 'scripts/qwen/qwen_omni_c4.py', 'scripts/sms_wsj/mix_slurp_v1.py',
        'data/slurp/manifests/split_v2_seed2026/devel_recordings.csv', 'data/slurp/manifests/split_v2_seed2026/train_recordings.csv',
        'docs/lora_slu_plan_v2.md']


def items(name):
    def rows(split):
        R = sorted(csv.DictReader(open(RV.MAN / f'{split}_recordings.csv', newline='')), key=lambda r: r['file'])
        return [dict(file=r['file'], slurp_id=int(r['slurp_id']), sentence=r['sentence'], intent=r['intent'], entities=json.loads(r['values'])) for r in R]
    if name == 'devel_clean':
        return [dict(r, audio='clean', split='devel') for r in rows('devel')]
    if name == 'devel_rev':
        return [dict(r, audio='rev', split='devel') for r in rows('devel')]
    if name in ('test_clean', 'test_rev'):
        T = rows('test'); assert len(T) == 1183
        return [dict(r, audio='clean' if name == 'test_clean' else 'rev', split='test') for r in T]
    tr = rows('train'); assert len(tr) == 3513
    return [dict(r, audio='clean', split='train') for r in tr] + [dict(r, audio='rev', split='train') for r in tr]


def load_audio(it, rv_assign, allow_compute):
    if it['audio'] == 'clean':
        x, sr = sf.read(RV.SLURP / it['file'], dtype='float32'); assert sr == SR and x.ndim == 1
        return x, 'file'
    p = REVDIR[it['split']] / f"{Path(it['file']).stem}.wav"
    if p.exists():
        x, sr = sf.read(p, dtype='float32'); assert sr == SR and x.ndim == 1
        return x, 'file'
    assert allow_compute, f'missing reverb audio {p}'
    r = rv_assign[it['file']]
    return RV.reverb(r['file'], r['rir_set'], r['rir_example'], r['rir_pos'])[0], 'computed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('p2', 'whisper_t2', 'whisper'), required=True)
    ap.add_argument('--items', choices=('devel_clean', 'devel_rev', 'train_all', 'test_clean', 'test_rev'), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1); ap.add_argument('--dry-run', type=int, default=0)
    a = ap.parse_args()
    assert not (a.items == 'train_all' and a.mode != 'whisper'), 'train items are only transcribed (step 2)'
    IT = items(a.items); n = len(IT); IT = IT[a.part * n // a.nparts:(a.part + 1) * n // a.nparts]
    rv = {s: {r['file']: r for r in RV.assignment(s)} for s in {it['split'] for it in IT if it['audio'] == 'rev'}}
    t0 = time.perf_counter(); W, src = {}, {}
    for it in IT:
        k = (it['audio'], it['file']); W[k], src[k] = load_audio(it, rv.get(it['split'], {}), a.dry_run > 0)
    print('AUDIO_LOADED', len(W), 'of', n, 'part', a.part, '/', a.nparts, 'sources', {s: list(src.values()).count(s) for s in set(src.values())},
          'hours', round(sum(len(x) for x in W.values()) / SR / 3600, 3), 'sec', round(time.perf_counter() - t0, 1), flush=True)
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()},
                mode=a.mode, items=a.items, part=a.part, nparts=a.nparts, n_part=len(IT), n_set=n)
    hub = Path(os.environ['HF_HOME']) / 'hub'
    qdir = hub / 'models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION; wdir = hub / 'models--openai--whisper-large-v3/snapshots' / WHISPER_REV
    assert qdir.is_dir() and (wdir / 'model.safetensors').exists()
    from transformers import Qwen2_5OmniProcessor, WhisperProcessor
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True) if a.mode != 'whisper' else None
    wp = WhisperProcessor.from_pretrained(str(wdir), local_files_only=True) if a.mode != 'p2' else None
    if a.dry_run:
        for it in IT[:a.dry_run]:
            x = W[(it['audio'], it['file'])]
            if a.mode == 'p2':
                _, b = build(qp, PROMPT_P2, x); print('DRY P2', it['audio'], it['file'], list(b['input_ids'].shape), flush=True)
            else:
                f = wp(x, sampling_rate=SR, return_tensors='pt').input_features; print('DRY W', it['audio'], it['file'], list(f.shape), flush=True)
                if a.mode == 'whisper_t2':
                    _, b = build(qp, [it['sentence'].strip() + '\n', PROMPT_T2]); print('DRY T2', list(b['input_ids'].shape), flush=True)
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, WhisperForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=False); prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    (a.out / 'config.json').write_text(json.dumps(dict(spec='docs/lora_slu_plan_v2.md; docs/lora_v2_runs.md §10', model_commit=REVISION, whisper_revision=WHISPER_REV,
                                                       prompt_P2=PROMPT_P2, prompt_T2=PROMPT_T2, decoding=dict(do_sample=False, thinker_max_new_tokens=256),
                                                       whisper_generate=dict(language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128),
                                                       **prov), indent=1, ensure_ascii=False) + '\n')
    t0 = time.perf_counter()
    if a.mode != 'whisper':
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
        model.disable_talker()
    if a.mode != 'p2':
        wm = WhisperForConditionalGeneration.from_pretrained(str(wdir), torch_dtype=torch.float16, local_files_only=True).to('cuda').eval()
    torch.cuda.synchronize(); print('LOAD', round(time.perf_counter() - t0, 1), flush=True)

    def qgen(batch):
        batch = batch.to(model.device)
        for k, v in batch.items():
            if torch.is_floating_point(v):
                batch[k] = v.to(model.dtype)
        torch.cuda.synchronize(); s0 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
        torch.cuda.synchronize()
        return qp.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0], time.perf_counter() - s0, int(ids.shape[1] - batch['input_ids'].shape[1])

    def rec(cond, it, text, raw, sec, nt, **kw):
        return dict(condition=cond, file=it['file'], audio=it['audio'], slurp_id=it['slurp_id'], ref_sentence=it['sentence'], ref_intent=it['intent'],
                    ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1] if cond != 'W' else None, seconds=sec, new_tokens=nt, **kw)
    if a.mode == 'p2':
        d = a.out / 'P2'; d.mkdir(); tc = time.perf_counter()
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in IT:
                text, b = build(qp, PROMPT_P2, W[(it['audio'], it['file'])]); raw, sec, nt = qgen(b)
                fh.write(json.dumps(rec('P2', it, text, raw, sec, nt), ensure_ascii=False) + '\n'); fh.flush()
        print('COND_DONE P2', round(time.perf_counter() - tc, 1), flush=True)
    else:
        d = a.out / 'W'; d.mkdir(); tc = time.perf_counter(); asr = {}
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in IT:
                f = wp(W[(it['audio'], it['file'])], sampling_rate=SR, return_tensors='pt').input_features.to('cuda', torch.float16)
                torch.cuda.synchronize(); s0 = time.perf_counter()
                with torch.inference_mode():
                    ids = wm.generate(f, language='en', task='transcribe', num_beams=1, do_sample=False, max_new_tokens=128)
                torch.cuda.synchronize(); txt = wp.batch_decode(ids, skip_special_tokens=True)[0]; asr[(it['audio'], it['file'])] = txt
                fh.write(json.dumps(rec('W', it, None, txt, time.perf_counter() - s0, int(ids.shape[1])), ensure_ascii=False) + '\n'); fh.flush()
        print('COND_DONE W', round(time.perf_counter() - tc, 1), flush=True)
        if a.mode == 'whisper_t2':
            d = a.out / 'T'; d.mkdir(); tc = time.perf_counter()
            with open(d / 'raw.jsonl', 'w') as fh:
                for it in IT:
                    t = asr[(it['audio'], it['file'])]; text, b = build(qp, [t.strip() + '\n', PROMPT_T2]); raw, sec, nt = qgen(b)
                    fh.write(json.dumps(rec('T', it, text, raw, sec, nt, asr_text=t), ensure_ascii=False) + '\n'); fh.flush()
            print('COND_DONE T', round(time.perf_counter() - tc, 1), flush=True)
    print('PEAK_GPU_GIB', round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True); print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
