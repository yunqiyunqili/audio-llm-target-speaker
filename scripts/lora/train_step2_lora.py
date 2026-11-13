"""Step 2 Whisper+T2-LoRA; frozen docs/lora_v2_runs.md §15.
Uses the exact Step-1 7026-item order, optimizer, loss normalization, checkpoint
schedule and segmented resume checks. Text-only input from existing Whisper train
outputs; SLURP gold JSON target (including gold transcript), frozen T2.
No audio encoding or ASR rerun. --pilot is isolated; never used as a trained run.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/lora'))
from qwen_omni_single import REVISION, SR  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_T2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402
from train_text_lora import lr_at, target_json  # noqa: E402
from train_audio_lora import load_train as audio_train_items  # noqa: E402
from qwen_omni_c7w_f1 import WHISPER_REV  # noqa: E402

MAN = PROJ / 'data/slurp/manifests/split_v2_seed2026'
SLURP = Path('<SHARED_ROOT>/data/slurp/audio/slurp_real')
REV = Path('<SHARED_ROOT>/data/lora_v1/train_rev_si284')
TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'
SEED, MB, ACC, EPOCHS = 2026, 4, 4, 3
SOFT_STOP, HARD_STOP = 24 * 60, 27 * 60
ASR_ROOT = PROJ / 'outputs/lora_base_audio/whisper_train_all'
DEPS = ['scripts/lora/train_step2_lora.py', 'scripts/lora/train_step2_lora.sbatch',
        'scripts/lora/train_audio_lora.py', 'scripts/lora/train_text_lora.py',
        'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'docs/lora_slu_plan_v2.md']
DATA_SHA = {}
T0 = time.time()


def load_train():
    expected = audio_train_items()  # existing split-leakage and train_si284 manifest audit
    raw = {}
    paths = sorted(ASR_ROOT.glob('*/W/raw.jsonl')); assert len(paths) == 6
    for path in paths:
        cfg_path = path.parents[1] / 'config.json'
        cfg = json.loads(cfg_path.read_text())
        assert cfg['whisper_revision'] == WHISPER_REV and not cfg['git_dirty_files']
        assert cfg['prompt_T2'] == PROMPT_T2
        for p in [path, cfg_path]: DATA_SHA[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            r = json.loads(line); k = r['audio'] + '|' + r['file']
            assert r['condition'] == 'W' and k not in raw and k in expected, k
            assert (r['ref_sentence'], r['ref_intent'], r['ref_entities']) == expected[k][1], k
            assert isinstance(r['raw_output'], str)
            raw[k] = (r['raw_output'], expected[k][1])
    assert set(raw) == set(expected) and len(raw) == 7026
    print('DATA_AUDIT_PASS', len(raw), 'unique train audio keys; gold targets verified', flush=True)
    return raw


def schedule(keys):
    steps = []
    for e in range(EPOCHS):
        ids = sorted(keys); random.Random(SEED + e).shuffle(ids)
        micro = [ids[i:i + MB] for i in range(0, len(ids), MB)]
        for j in range(0, len(micro), ACC):
            steps.append(dict(epoch=e, micro=micro[j:j + ACC]))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, required=True); ap.add_argument('--segment', type=int, required=True)
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--pilot', type=int, default=0); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    IT = load_train(); STEPS = schedule(IT); TOTAL = len(STEPS); WARM = round(0.03 * TOTAL)
    per_epoch = TOTAL // EPOCHS; CK = [round(per_epoch * h / 2) for h in range(1, 2 * EPOCHS + 1)]
    assert TOTAL == 1320 and per_epoch == 440 and WARM == 40 and CK == [220, 440, 660, 880, 1100, 1320], (TOTAL, WARM, CK)
    assert sum(len(m) for m in STEPS[per_epoch - 1]['micro']) == 2
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True); tok = qp.tokenizer
    assert tok.padding_side == 'right'
    cache = {}

    def enc(k):
        if k not in cache:
            asr, meta = IT[k]
            text, b = build(qp, [asr.strip() + '\n', PROMPT_T2])
            p = tok(text, add_special_tokens=False)['input_ids']
            assert p == b['input_ids'][0].tolist(), k
            t = target_json(*meta) + '<|im_end|>'
            o = tok(t, add_special_tokens=False, return_offsets_mapping=True)
            cut = t.index('"entities"')
            cache[k] = (p, o['input_ids'], [st >= cut for st, _ in o['offset_mapping']])
        return cache[k]
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS}, lr=a.lr, segment=a.segment, pilot=a.pilot, data_sha256=DATA_SHA)
    run = a.run_dir; state_p = run / 'state.json'
    if a.segment == 1:
        assert not run.exists(), run
        start = 0
    else:
        prev = json.load(open(run / f'provenance_seg{a.segment-1}.json'))
        assert prev['data_sha256'] == DATA_SHA and prev['lr'] == a.lr
        st = json.load(open(state_p)); assert st['segment'] == a.segment - 1 and not st.get('finished'), st; start = st['resume_step']
    print('PLAN', json.dumps(dict(total_steps=TOTAL, warmup=WARM, checkpoints=CK, start_step=start + 1, first_batch=STEPS[start]['micro'][0],
                                  lr_first=lr_at(a.lr, start, TOTAL, WARM), lr_peak=lr_at(a.lr, WARM - 1, TOTAL, WARM), lr_last=lr_at(a.lr, TOTAL - 1, TOTAL, WARM),
                                  soft_stop_s=SOFT_STOP, hard_stop_s=HARD_STOP)), flush=True)
    if a.dry_run:
        for k in sorted(IT):
            p, t, ent = enc(k)
            assert tok.decode(t) == target_json(*IT[k][1]) + '<|im_end|>', k
        assert len(cache) == 7026
        print('ALL_INPUT_IDS_MATCH_INFERENCE', len(cache), flush=True)
        print('TOKEN_LENGTH_MAX', max(len(p)+len(t) for p,t,e in cache.values()), flush=True)
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen2_5OmniForConditionalGeneration
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / f'provenance_seg{a.segment}.json').exists()
    prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    (run / f'provenance_seg{a.segment}.json').write_text(json.dumps(prov, indent=1) + '\n')
    torch.manual_seed(SEED)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker(); th = model.thinker
    for q in model.parameters():
        q.requires_grad_(False)
    th.config.use_cache = False; th.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    pm = get_peft_model(th, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_RE, bias='none'))
    P = [q for q in pm.parameters() if q.requires_grad]
    assert len({n.rsplit('.lora_', 1)[0] for n, q in pm.named_parameters() if q.requires_grad}) == 112
    assert not any('audio_tower' in n for n, q in pm.named_parameters() if q.requires_grad)
    opt = torch.optim.AdamW(P, lr=a.lr, weight_decay=0.0)
    dev = torch.device('cuda'); print('LOADED', round(time.time() - T0, 1), 'GiB', round(torch.cuda.memory_allocated() / 2**30, 2),
                                      'load_peak_GiB', round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
    torch.cuda.reset_peak_memory_stats()  # from here on, peaks are training-phase only (docs/lora_v2_runs.md §17)

    def lsum():
        return float(sum(q.detach().double().abs().sum() for q in P))

    def osum():
        st = opt.state_dict()['state']
        return float(sum(v.double().abs().sum() for s in st.values() for k, v in s.items() if torch.is_tensor(v)))

    def micro_loss(keys):
        B = [enc(k) for k in keys]; ml = max(len(p) + len(t) for p,t,e in B)
        pad = tok.pad_token_id
        ids = torch.full((len(B), ml), pad, dtype=torch.long); lab = torch.full_like(ids, -100); att = torch.zeros_like(ids); em = torch.zeros_like(ids, dtype=torch.bool)
        for k, (p, t, e) in enumerate(B):
            seq = p + t; ids[k, :len(seq)] = torch.tensor(seq); att[k, :len(seq)] = 1; lab[k, len(p):len(seq)] = torch.tensor(t); em[k, len(p):len(seq)] = torch.tensor(e)
        ids, lab, att, em = ids.to(dev), lab.to(dev), att.to(dev), em.to(dev)
        lg = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
        y = lab[:, 1:]; m = y != -100; me = em[:, 1:][m]
        ce = torch.nn.functional.cross_entropy(lg[m].float(), y[m], reduction='none')
        assert torch.isfinite(ce).all(), 'nonfinite loss: STOP, no automatic retry'
        return ce.mean(), float(ce.detach()[me].sum()), int(me.sum()), sum(len(p) for p,t,e in B)

    def opt_step(s):
        stp = STEPS[s]; lr = lr_at(a.lr, s, TOTAL, WARM)
        for g in opt.param_groups:
            g['lr'] = lr
        n = len(stp['micro']); tot = 0.0; es = 0.0; en = 0
        for mb in stp['micro']:
            loss, e_s, e_n, _ = micro_loss(mb); (loss / n).backward(); tot += float(loss) / n; es += e_s; en += e_n
        gn = float(torch.nn.utils.clip_grad_norm_(P, 1.0)); assert math.isfinite(gn), 'nonfinite gradient'; opt.step(); opt.zero_grad(set_to_none=True)
        return dict(step=s + 1, epoch=round((s + 1) / per_epoch, 4), lr=lr, loss=tot, entity_loss=es / max(en, 1), entity_tokens=en, grad_norm=gn,
                    batch_first=stp['micro'][0][:2])
    if a.segment > 1:
        rp = run / f'resume_step{start}'
        set_peft_model_state_dict(pm, load_file(str(rp / 'adapter_model.safetensors')))
        R = torch.load(rp / 'resume.pt', weights_only=False)
        opt.load_state_dict(R['optimizer']); torch.set_rng_state(R['rng_cpu']); torch.cuda.set_rng_state(R['rng_cuda'])
        ref = json.load(open(rp / 'reference.json'))
        chk = dict(global_step=R['global_step'] == start, lora_checksum=abs(lsum() - R['lora_checksum']) <= 1e-9 * max(1.0, R['lora_checksum']),
                   optimizer_state=abs(osum() - R['opt_checksum']) <= 1e-9 * max(1.0, R['opt_checksum']) and
                   all(int(s['step']) == start for s in opt.state_dict()['state'].values()),
                   next_lr=all(abs(lr_at(a.lr, start + k, TOTAL, WARM) - ref['steps'][k]['lr']) < 1e-15 for k in range(len(ref['steps']))),
                   data_order=all(STEPS[start + k]['micro'] == ref['micro'][k] for k in range(len(ref['steps']))))
        out = dict(segment=a.segment, resume_step=start, exact_checks=chk, passed=all(chk.values()))
        if not out['passed']:
            (run / f'resume_check_seg{a.segment}.json').write_text(json.dumps(out, indent=1) + '\n'); print('RESUME_CHECK_FAIL', json.dumps(out), flush=True); sys.exit(3)
        pm.train(); comp = []
        for k in range(len(ref['steps'])):
            r = opt_step(start + k); c = lsum()
            comp.append(dict(step=r['step'], loss=r['loss'], ref_loss=ref['steps'][k]['loss'], lora_checksum=c, ref_lora_checksum=ref['steps'][k]['lora_checksum'],
                             rel_loss_diff=abs(r['loss'] - ref['steps'][k]['loss']) / max(1e-12, abs(ref['steps'][k]['loss'])),
                             rel_checksum_diff=abs(c - ref['steps'][k]['lora_checksum']) / ref['steps'][k]['lora_checksum']))
            with open(run / 'log.jsonl', 'a') as fh:
                fh.write(json.dumps(dict(r, seconds=None, segment=a.segment)) + '\n')
            if r['step'] in CK:
                pm.save_pretrained(str(run / f"ckpt_step{r['step']}")); print('SAVED', r['step'], flush=True)
        out['reference_steps_comparison_report_only'] = comp
        (run / f'resume_check_seg{a.segment}.json').write_text(json.dumps(out, indent=1) + '\n'); print('RESUME_CHECK_PASS', json.dumps(out), flush=True)
        s = start + len(ref['steps'])
    else:
        s = 0
    pm.train(); stop = TOTAL if not a.pilot else a.pilot; times = []
    while s < stop:
        el = time.time() - T0
        if el > HARD_STOP:
            print('TIME_GUARD_ABORT', s, flush=True); sys.exit(4)
        if el > SOFT_STOP and not a.pilot:
            break
        t1 = time.perf_counter(); r = opt_step(s); r['seconds'] = time.perf_counter() - t1; r['segment'] = a.segment; times.append(r['seconds'])
        with open(run / 'log.jsonl', 'a') as fh:
            fh.write(json.dumps(r) + '\n')
        if (s + 1) % 10 == 0 or s == 0:
            print('STEP', json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'batch_first'}), 'peak_GiB', round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
        s += 1
        if s in CK and not a.pilot:
            pm.save_pretrained(str(run / f'ckpt_step{s}')); print('SAVED', s, flush=True)
    finished = s >= TOTAL
    if not finished:
        rp = run / f'resume_step{s}'; rp.mkdir(exist_ok=False); t2 = time.perf_counter()
        pm.save_pretrained(str(rp)); assert (rp / 'adapter_model.safetensors').exists()
        torch.save(dict(global_step=s, optimizer=opt.state_dict(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
                        lora_checksum=lsum(), opt_checksum=osum()), rp / 'resume.pt')
        ref = dict(micro=[STEPS[s + k]['micro'] for k in range(min(2, TOTAL - s))], steps=[])
        for k in range(len(ref['micro'])):
            r = opt_step(s + k); ref['steps'].append(dict(step=r['step'], lr=r['lr'], loss=r['loss'], lora_checksum=lsum()))
        (rp / 'reference.json').write_text(json.dumps(ref, indent=1) + '\n')
        print('RESUME_STATE_SAVED', s, round(time.perf_counter() - t2, 1), 's', json.dumps(ref['steps']), flush=True)
    if not a.pilot:
        state_p.write_text(json.dumps(dict(segment=a.segment, resume_step=None if finished else s, finished=finished, job=os.environ.get('SLURM_JOB_ID')), indent=1) + '\n')
    peak = torch.cuda.max_memory_allocated() / 2**30
    med = sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else None
    print('SEGMENT_DONE', a.segment, 'start_step', start + 1, 'last_step', s, 'finished', finished, 'median_step_s', med and round(med, 3),
          'train_peak_alloc_GiB', round(peak, 2), 'train_peak_reserved_GiB', round(torch.cuda.max_memory_reserved() / 2**30, 2), 'wall_s', round(time.time() - T0, 1), flush=True)


if __name__ == '__main__':
    main()
