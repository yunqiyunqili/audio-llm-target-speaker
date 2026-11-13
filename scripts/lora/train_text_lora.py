"""Step-0 text LoRA training (docs/lora_slu_plan_v2.md + revision 1, frozen). TRAIN split only.
Data: 2,363 unique train sentences; input = build(processor, [sentence + '\\n', PROMPT_T2]); target = SLURP gold JSON (T2 format)
+ '<|im_end|>'; loss on target tokens only. Entity-part token loss (tokens from '"entities"' on) logged for observation only.
Config: LoRA r16/alpha32/dropout0.05 on thinker LM q/k/v/o (regex, 112 modules); bf16; grad checkpointing; AdamW (wd 0);
grad clip 1.0; micro-batch 4 x accumulation 4 = effective 16; 3 epochs = 444 optimizer steps (148 / epoch; the last step of an
epoch has 3 micro-batches / 11 samples); warmup 13 steps then cosine to 0; seed 2026.
Epoch e order: sorted slurp_ids shuffled by random.Random(2026 + e). Checkpoints every 0.5 epoch: global steps 74,148,...,444.
Segments: 1 = steps 1-222, 2 = steps 223-444. Segment 1 saves resume state at step 222, then runs 2 REFERENCE steps (not saved)
recording lr / batch ids / losses / LoRA checksum. Segment 2 verifies global step, next lr, optimizer state, data order (exact;
failure -> exit 3, no training) and then compares the 2 reference steps (reported only).
Usage: python scripts/lora/train_text_lora.py --lr LR --segment {1,2} --run-dir DIR [--dry-run]
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

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen'))
from qwen_omni_single import REVISION  # noqa: E402
from qwen_omni_c1_p2t2 import PROMPT_T2  # noqa: E402
from qwen_omni_c6c7 import build  # noqa: E402

MAN = PROJ / 'data/slurp/manifests/split_v2_seed2026'
TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'
SEED, MB, ACC, EPOCHS = 2026, 4, 4, 3
DEPS = ['scripts/lora/train_text_lora.py', 'scripts/lora/train_text_lora.sbatch', 'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_single.py', 'docs/lora_slu_plan_v2.md']
T0 = time.time()


def target_json(sentence, intent, ents):
    return json.dumps(dict(transcript=sentence, intent=intent, entities=[dict(type=t, value=v) for t, v in ents]), ensure_ascii=False)


def load_train():
    tr = list(csv.DictReader(open(MAN / 'train_recordings.csv', newline='')))
    dv = list(csv.DictReader(open(MAN / 'devel_recordings.csv', newline=''))); te = list(csv.DictReader(open(MAN / 'test_recordings.csv', newline='')))
    by = {}
    for r in tr:
        by.setdefault(int(r['slurp_id']), (r['sentence'], r['intent'], json.loads(r['values'])))
    S = {v[0] for v in by.values()}
    leak = [len(S & {r['sentence'] for r in dv}), len(S & {r['sentence'] for r in te}), len(set(by) & {int(r['slurp_id']) for r in dv + te}),
            len({r['usrid'] for r in tr} & {r['usrid'] for r in dv + te})]
    assert not any(leak), leak
    return by


def schedule():
    """List of optimizer steps: each = list of micro-batches (lists of slurp_ids); plus epoch index."""
    steps = []
    for e in range(EPOCHS):
        ids = sorted(BY); random.Random(SEED + e).shuffle(ids)
        micro = [ids[i:i + MB] for i in range(0, len(ids), MB)]
        for j in range(0, len(micro), ACC):
            steps.append(dict(epoch=e, micro=micro[j:j + ACC]))
    return steps


def lr_at(base, s, total, warm):
    """lr for optimizer step index s (0-based)."""
    if s < warm:
        return base * (s + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm)))


BY = None


def main():
    global BY
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, required=True); ap.add_argument('--segment', type=int, choices=(1, 2), required=True)
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    BY = load_train(); STEPS = schedule(); TOTAL = len(STEPS); WARM = round(0.03 * TOTAL)
    per_epoch = TOTAL // EPOCHS; CK = [round(per_epoch * h / 2) for h in range(1, 2 * EPOCHS + 1)]; MID = CK[2]
    assert TOTAL == 444 and per_epoch == 148 and CK == [74, 148, 222, 296, 370, 444], (TOTAL, CK)
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True); tok = qp.tokenizer
    cache = {}

    def enc(i):
        if i not in cache:
            s, it, en = BY[i]
            text, _ = build(qp, [s.strip() + '\n', PROMPT_T2]); p = tok(text, add_special_tokens=False)['input_ids']
            tgt = target_json(s, it, en) + '<|im_end|>'; o = tok(tgt, add_special_tokens=False, return_offsets_mapping=True)
            cut = tgt.index('"entities"'); ent = [st >= cut for st, _ in o['offset_mapping']]
            cache[i] = (p, o['input_ids'], ent)
        return cache[i]
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS}, lr=a.lr, segment=a.segment)
    rng_s = range(0, MID) if a.segment == 1 else range(MID, TOTAL)
    print('PLAN', json.dumps(dict(total_steps=TOTAL, warmup=WARM, checkpoints=CK, segment_steps=[rng_s.start + 1, rng_s.stop],
                                  first_batch=STEPS[rng_s.start]['micro'][0], lr_first=lr_at(a.lr, rng_s.start, TOTAL, WARM),
                                  lr_peak=lr_at(a.lr, WARM - 1, TOTAL, WARM), lr_last=lr_at(a.lr, TOTAL - 1, TOTAL, WARM),
                                  last_step_of_epoch_samples=sum(len(m) for m in STEPS[per_epoch - 1]['micro']))), flush=True)
    p, t, e = enc(STEPS[0]['micro'][0][0]); print('ENC', len(p), len(t), sum(e), tok.decode([x for x, f in zip(t, e) if f])[:80], flush=True)
    if a.dry_run:
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen2_5OmniForConditionalGeneration
    run = a.run_dir; run.mkdir(parents=True, exist_ok=True)
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
    opt = torch.optim.AdamW(P, lr=a.lr, weight_decay=0.0)
    dev = torch.device('cuda')

    def lsum():
        return float(sum(q.detach().double().abs().sum() for q in P))

    def osum():
        st = opt.state_dict()['state']
        return float(sum(v.double().abs().sum() for s in st.values() for k, v in s.items() if torch.is_tensor(v)))

    def micro_loss(ids_list):
        B = [enc(i) for i in ids_list]; ml = max(len(p) + len(t) for p, t, _ in B)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(B), ml), pad, dtype=torch.long); lab = torch.full_like(ids, -100); att = torch.zeros_like(ids); em = torch.zeros_like(ids, dtype=torch.bool)
        for k, (p, t, e) in enumerate(B):
            s = p + t; ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1; lab[k, len(p):len(s)] = torch.tensor(t); em[k, len(p):len(s)] = torch.tensor(e)
        ids, lab, att, em = ids.to(dev), lab.to(dev), att.to(dev), em.to(dev)
        lg = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]; y = lab[:, 1:]; m = y != -100; me = em[:, 1:][m]
        ce = torch.nn.functional.cross_entropy(lg[m].float(), y[m], reduction='none')
        loss = ce.mean()
        return loss, float(ce.detach()[me].sum()), int(me.sum())

    def opt_step(s):
        st = STEPS[s]; lr = lr_at(a.lr, s, TOTAL, WARM)
        for g in opt.param_groups:
            g['lr'] = lr
        n = len(st['micro']); tot = 0.0; es = 0.0; en = 0
        for mb in st['micro']:
            loss, e_s, e_n = micro_loss(mb); (loss / n).backward(); tot += float(loss) / n; es += e_s; en += e_n
        gn = float(torch.nn.utils.clip_grad_norm_(P, 1.0)); opt.step(); opt.zero_grad(set_to_none=True)
        return dict(step=s + 1, epoch=round((s + 1) / per_epoch, 4), lr=lr, loss=tot, entity_loss=es / max(en, 1), entity_tokens=en, grad_norm=gn,
                    batch_first=st['micro'][0][:2])

    res_path = run / 'resume_step222'
    if a.segment == 2:
        set_peft_model_state_dict(pm, load_file(str(run / f'ckpt_step{MID}' / 'adapter_model.safetensors')))
        R = torch.load(res_path / 'resume.pt', weights_only=False)
        opt.load_state_dict(R['optimizer']); torch.set_rng_state(R['rng_cpu']); torch.cuda.set_rng_state(R['rng_cuda'])
        ref = json.load(open(res_path / 'reference.json'))
        chk = dict(global_step=R['global_step'] == MID, lora_checksum=abs(lsum() - R['lora_checksum']) <= 1e-9 * max(1.0, R['lora_checksum']),
                   optimizer_state=abs(osum() - R['opt_checksum']) <= 1e-9 * max(1.0, R['opt_checksum']) and
                   all(int(s['step']) == MID for s in opt.state_dict()['state'].values()),
                   next_lr=all(abs(lr_at(a.lr, MID + k, TOTAL, WARM) - ref['steps'][k]['lr']) < 1e-15 for k in range(2)),
                   data_order=all(STEPS[MID + k]['micro'] == ref['micro'][k] for k in range(2)))
        out = dict(exact_checks=chk, passed=all(chk.values()))
        if not out['passed']:
            (run / 'resume_check.json').write_text(json.dumps(out, indent=1) + '\n'); print('RESUME_CHECK_FAIL', json.dumps(out), flush=True); sys.exit(3)
        pm.train(); comp = []
        for k in range(2):
            r = opt_step(MID + k); c = lsum()
            comp.append(dict(step=r['step'], loss=r['loss'], ref_loss=ref['steps'][k]['loss'], lora_checksum=c, ref_lora_checksum=ref['steps'][k]['lora_checksum'],
                             rel_loss_diff=abs(r['loss'] - ref['steps'][k]['loss']) / max(1e-12, abs(ref['steps'][k]['loss'])),
                             rel_checksum_diff=abs(c - ref['steps'][k]['lora_checksum']) / ref['steps'][k]['lora_checksum']))
            with open(run / 'log.jsonl', 'a') as fh:
                fh.write(json.dumps(dict(**r, seconds=None, segment=2)) + '\n')
        out['reference_steps_comparison_report_only'] = comp
        (run / 'resume_check.json').write_text(json.dumps(out, indent=1) + '\n'); print('RESUME_CHECK_PASS', json.dumps(out), flush=True)
        start = MID + 2
    else:
        start = 0
    pm.train()
    for s in range(start, rng_s.stop):
        if time.time() - T0 > 27 * 60:
            print('TIME_GUARD_ABORT', s, flush=True); sys.exit(4)
        t1 = time.perf_counter(); r = opt_step(s); r['seconds'] = time.perf_counter() - t1; r['segment'] = a.segment
        with open(run / 'log.jsonl', 'a') as fh:
            fh.write(json.dumps(r) + '\n')
        if (s + 1) % 10 == 0 or s == start:
            print('STEP', json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'batch_first'}), flush=True)
        if s + 1 in CK:
            pm.save_pretrained(str(run / f'ckpt_step{s + 1}')); print('SAVED', s + 1, flush=True)
    if a.segment == 1:
        res_path.mkdir(exist_ok=False)
        torch.save(dict(global_step=MID, optimizer=opt.state_dict(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
                        lora_checksum=lsum(), opt_checksum=osum()), res_path / 'resume.pt')
        ref = dict(micro=[STEPS[MID + k]['micro'] for k in range(2)], steps=[])
        for k in range(2):
            r = opt_step(MID + k); ref['steps'].append(dict(step=r['step'], lr=r['lr'], loss=r['loss'], lora_checksum=lsum()))
        (res_path / 'reference.json').write_text(json.dumps(ref, indent=1) + '\n')
        print('RESUME_STATE_SAVED', json.dumps(ref['steps']), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print('SEGMENT_DONE', a.segment, 'peak_alloc_GiB', round(peak, 2), 'wall_s', round(time.time() - T0, 1), flush=True)


if __name__ == '__main__':
    main()
