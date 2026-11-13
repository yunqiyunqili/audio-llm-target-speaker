"""Step-0 (text LoRA) PILOT for docs/lora_slu_plan_v2.md (frozen). Measures speed and GPU memory only; NOT a training run.
Data: split_v2 TRAIN only (2,363 unique slurp_id sentences, shuffled with seed 2026). Input = build(processor,
[sentence + '\\n', PROMPT_T2]) (frozen T2, same as C0); target = SLURP gold JSON in the T2 output format + '<|im_end|>';
loss on target tokens only. LoRA r16/alpha32/dropout0.05 on thinker LM q/k/v/o (regex, 112 modules); audio tower frozen,
talker disabled; bf16; gradient checkpointing; AdamW lr 1e-4; grad accumulation 16.
Phases: A) 200 micro-steps at micro-batch 1; B) 40 micro-steps at micro-batch 4 (right padding); C) save adapter +
optimizer state, report sizes; D) generate on 3 devel SELECTION-half sentences, report parse status (format sanity only).
Wall-clock guard: phases A/B stop early if the job would exceed ~24 min. --dry-run: CPU, tokenizer only, data checks.
Usage: python scripts/lora/text_lora_pilot.py --out DIR [--dry-run]
"""
import argparse
import csv
import hashlib
import json
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
from slu_normalize import parse  # noqa: E402

MAN = PROJ / 'data/slurp/manifests/split_v2_seed2026'
TARGET_RE = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'
DEPS = ['scripts/lora/text_lora_pilot.py', 'scripts/lora/text_lora_pilot.sbatch', 'scripts/qwen/qwen_omni_c6c7.py', 'scripts/qwen/qwen_omni_c1_p2t2.py',
        'scripts/qwen/qwen_omni_single.py', 'docs/lora_slu_plan_v2.md']
T_START = time.time()


def target_json(sentence, intent, ents):
    return json.dumps(dict(transcript=sentence, intent=intent, entities=[dict(type=t, value=v) for t, v in ents]), ensure_ascii=False)


def load_data():
    tr = list(csv.DictReader(open(MAN / 'train_recordings.csv', newline='')))
    dv = list(csv.DictReader(open(MAN / 'devel_recordings.csv', newline='')))
    te = list(csv.DictReader(open(MAN / 'test_recordings.csv', newline='')))
    by = {}
    for r in tr:
        by.setdefault(int(r['slurp_id']), (r['sentence'], r['intent'], json.loads(r['values'])))
    leak = dict(sentence_devel=len({v[0] for v in by.values()} & {r['sentence'] for r in dv}), sentence_test=len({v[0] for v in by.values()} & {r['sentence'] for r in te}),
                id_devel=len(set(by) & {int(r['slurp_id']) for r in dv}), id_test=len(set(by) & {int(r['slurp_id']) for r in te}),
                speaker_devel=len({r['usrid'] for r in tr} & {r['usrid'] for r in dv}), speaker_test=len({r['usrid'] for r in tr} & {r['usrid'] for r in te}))
    assert not any(leak.values()), leak
    ids = sorted(by); random.Random(2026).shuffle(ids)
    # devel selection half (docs/lora_slu_plan_v2.md §3.2): first 191 shuffled devel slurp_ids
    did = sorted({int(r['slurp_id']) for r in dv}); random.Random(2026).shuffle(did); sel = did[:len(did) // 2]
    dsent = {int(r['slurp_id']): r['sentence'] for r in dv}
    return [dict(slurp_id=i, sentence=by[i][0], intent=by[i][1], entities=by[i][2]) for i in ids], [dsent[i] for i in sel[:3]], leak


def encode(tok, qp, it):
    text, _ = build(qp, [it['sentence'].strip() + '\n', PROMPT_T2])
    p = tok(text, add_special_tokens=False)['input_ids']; t = tok(target_json(it['sentence'], it['intent'], it['entities']) + '<|im_end|>', add_special_tokens=False)['input_ids']
    return p, t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    D, dev3, leak = load_data()
    qdir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    qp = Qwen2_5OmniProcessor.from_pretrained(str(qdir), local_files_only=True); tok = qp.tokenizer
    git = lambda *x: subprocess.run(['git', '-C', str(PROJ), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                node=os.environ.get('SLURMD_NODENAME', socket.gethostname()), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS}, python=sys.executable)
    enc = [encode(tok, qp, it) for it in D[:400]]
    L = [len(p) + len(t) for p, t in enc]; LT = [len(t) for _, t in enc]
    info = dict(n_train_sentences=len(D), leak_check=leak, first_ids=[it['slurp_id'] for it in D[:5]], tokens_total_mean=sum(L) / len(L), tokens_total_max=max(L),
                target_tokens_mean=sum(LT) / len(LT), dev3=dev3, example_target=target_json(D[0]['sentence'], D[0]['intent'], D[0]['entities']))
    # sanity: the prompt tokenisation equals the processor's (used at inference)
    _, b = build(qp, [D[0]['sentence'].strip() + '\n', PROMPT_T2]); assert b['input_ids'][0].tolist() == enc[0][0]
    print('DATA', json.dumps(info, ensure_ascii=False), flush=True)
    if a.dry_run:
        print('PROVENANCE', json.dumps(prov)); print('DRY_RUN_PASS', flush=True); return
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5OmniForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=False)
    prov['gpu'] = torch.cuda.get_device_name(0); print('PROVENANCE', json.dumps(prov), flush=True)
    torch.manual_seed(2026)
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(qdir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    th = model.thinker
    for p in model.parameters():
        p.requires_grad_(False)
    th.config.use_cache = False
    th.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    pm = get_peft_model(th, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_RE, bias='none'))
    tr_names = [n for n, p in pm.named_parameters() if p.requires_grad]
    n_mod = len({n.rsplit('.lora_', 1)[0] for n in tr_names}); n_par = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    assert n_mod == 112 and all('audio_tower' not in n for n in tr_names), (n_mod, tr_names[:3])
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    load_s = time.perf_counter() - t0; torch.cuda.synchronize(); base_mem = torch.cuda.memory_allocated() / 2**30
    print('LOAD', round(load_s, 1), 'lora_modules', n_mod, 'lora_params', n_par, 'mem_after_load_GiB', round(base_mem, 2), flush=True)
    dev = torch.device('cuda')

    def step(batch):
        ml = max(len(p) + len(t) for p, t in batch)
        ids = torch.full((len(batch), ml), tok.pad_token_id if tok.pad_token_id is not None else 0, dtype=torch.long); lab = torch.full_like(ids, -100); att = torch.zeros_like(ids)
        for i, (p, t) in enumerate(batch):
            s = p + t; ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1; lab[i, len(p):len(s)] = torch.tensor(t)
        ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
        out = pm(input_ids=ids, attention_mask=att, use_cache=False)
        lg = out.logits[:, :-1]; y = lab[:, 1:]; m = y != -100
        loss = torch.nn.functional.cross_entropy(lg[m].float(), y[m])
        (loss / 16).backward()
        return float(loss)

    res = dict(info=info, lora_modules=n_mod, lora_params=n_par, load_seconds=load_s, mem_after_load_GiB=base_mem, phases={})
    pm.train(); k = 0
    for name, mb, nsteps in (('A_mb1', 1, 200), ('B_mb4', 4, 40)):
        torch.cuda.reset_peak_memory_stats(); times = []; losses = []; acc = 0; opt_losses = []
        for s in range(nsteps):
            if time.time() - T_START > 24 * 60 - 240:
                print('TIME_GUARD', name, s, flush=True); break
            batch = []
            for _ in range(mb):
                if k >= len(enc):
                    enc.append(encode(tok, qp, D[k % len(D)]))
                batch.append(enc[k]); k += 1
            torch.cuda.synchronize(); s0 = time.perf_counter()
            losses.append(step(batch)); acc += 1
            if acc * mb >= 16:
                torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0); opt.step(); opt.zero_grad(set_to_none=True)
                opt_losses.append(sum(losses[-acc:]) / acc); acc = 0
            torch.cuda.synchronize(); times.append(time.perf_counter() - s0)
            if s % 20 == 0:
                print('STEP', name, s, round(times[-1], 3), round(losses[-1], 4), flush=True)
        st = sorted(times[5:]) or sorted(times)
        res['phases'][name] = dict(micro_batch=mb, micro_steps=len(times), median_s=st[len(st) // 2], mean_s=sum(st) / len(st),
                                   per_sample_s=(st[len(st) // 2]) / mb, peak_alloc_GiB=torch.cuda.max_memory_allocated() / 2**30,
                                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30, first_losses=losses[:5], last_losses=losses[-5:],
                                   optimizer_step_losses=opt_losses)
        print('PHASE', name, json.dumps(res['phases'][name]), flush=True)
    opt.zero_grad(set_to_none=True)
    ck = a.out / 'pilot_ckpt'; pm.save_pretrained(str(ck)); torch.save(opt.state_dict(), ck / 'optimizer.pt')
    res['saved'] = {p.name: p.stat().st_size for p in ck.iterdir()}
    pm.eval(); th.config.use_cache = True; gens = []
    for sent in dev3:
        text, b = build(qp, [sent.strip() + '\n', PROMPT_T2]); b = b.to(dev)
        with torch.inference_mode():
            ids = model.generate(**b, return_audio=False, thinker_max_new_tokens=256, do_sample=False, use_audio_in_video=False)
        raw = qp.batch_decode(ids[:, b['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        gens.append(dict(sentence=sent, raw_output=raw, parse_status=parse(raw)[1]))
    res['format_check_devel_selection3'] = gens; res['provenance'] = prov
    res['estimate_step0_one_config_hours'] = {nm: 3 * len(D) * ph['per_sample_s'] / 3600 for nm, ph in res['phases'].items()}
    (a.out / 'pilot.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print('GEN', json.dumps(gens, ensure_ascii=False), flush=True)
    print('ESTIMATE', json.dumps(res['estimate_step0_one_config_hours']), flush=True)
    print('PILOT_DONE', round(time.time() - T_START, 1), flush=True)


if __name__ == '__main__':
    main()
