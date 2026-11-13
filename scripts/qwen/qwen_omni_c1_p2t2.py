"""C1 prompt variants P2 / T2 on the same 100 DEVEL items as job 22010997 (configs/c1_ext_devel100.json).
  P2 = audio + P0 prompt (scripts/qwen/qwen_omni_c1_semantic.py PROMPT) without the line 'Use "entities": [] if there are
       none.' + the 3 sentences below; no examples.
  T2 = reference transcript (+ newline) as a text item, no audio, + T0 prompt with its first sentence replaced by
       "This text is a command to a voice assistant.", the same line removed, + the same 3 sentences.
"This command contains at least one entity." is valid only because every item of this filtered sample has >= 1 entity;
it must not be used for data that can contain entity-free commands.
Decoding identical to 22010997: greedy, max_new_tokens 256, batch 1, text only (talker disabled), use_audio_in_video=False.
Frozen files are only imported, not modified. Output format identical to qwen_omni_c1_ext.py (OUT/{P2,T2}/raw.jsonl,
summary.json) so scripts/qwen/rescore_slu_v2.py can score it. OUT/config.json records git commit, dirty state,
sha256 of this script and its dependencies, and the full prompts.
--dry-run N: CPU only, no weights (processor + prompt assembly + parse check).
Usage: python scripts/qwen/qwen_omni_c1_p2t2.py --audio-dir DIR --out DIR [--dry-run N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import REVISION, SR, SYSTEM  # noqa: E402
from qwen_omni_c1_semantic import PROMPT  # noqa: E402
from slu_normalize import parse  # noqa: E402

CFG = json.load(open(PROJ / 'configs/c1_ext_devel100.json'))
CONDS = ('P2', 'T2')
FIRST_AUDIO = 'This audio is a spoken command to a voice assistant.'
FIRST_TEXT = 'This text is a command to a voice assistant.'
EXTRA = ('Do not carry out or answer the command; only extract its meaning.\n'
         'This command contains at least one entity. List every entity from the command.\n'
         'Use only the allowed labels listed above.')
assert PROMPT.startswith(FIRST_AUDIO)
NONE_LINE = '\nUse "entities": [] if there are none.\n'
assert PROMPT.count(NONE_LINE) == 1
BASE = PROMPT.replace(NONE_LINE, '\n')      # removed from P2 and T2 only (decision 2026-11-05)
PROMPT_P2 = BASE + '\n' + EXTRA
PROMPT_T2 = FIRST_TEXT + BASE[len(FIRST_AUDIO):] + '\n' + EXTRA
DEPS = ['scripts/qwen/qwen_omni_c1_p2t2.py', 'scripts/qwen/qwen_omni_c1_semantic.py', 'scripts/qwen/qwen_omni_single.py',
        'scripts/qwen/slu_normalize.py', 'scripts/qwen/qwen_omni_c1_p2t2.sbatch', 'configs/c1_ext_devel100.json',
        'configs/slurp_label_inventory_train.json']


def provenance():
    git = lambda *a: subprocess.run(['git', '-C', str(PROJ), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty_files=git('status', '--porcelain', '--', *DEPS).splitlines(),
                sha256={p: hashlib.sha256((PROJ / p).read_bytes()).hexdigest() for p in DEPS if (PROJ / p).exists()})


def conversation(cond, it):
    if cond == 'T2':
        user = [{'type': 'text', 'text': it['sentence'] + '\n'}, {'type': 'text', 'text': PROMPT_T2}]
    else:
        user = [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': PROMPT_P2}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': user}]


def build(processor, cond, it, audio_dir):
    text = processor.apply_chat_template(conversation(cond, it), add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        assert len(text) == 1, len(text)
        text = text[0]
    if cond == 'T2':
        return text, processor(text=text, return_tensors='pt', padding=True)
    wav, sr = sf.read(audio_dir / it['file'], dtype='float32')
    assert sr == SR and wav.ndim == 1, (it['file'], sr, wav.shape)
    return text, processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--dry-run', type=int, default=0)
    args = ap.parse_args()
    from transformers import Qwen2_5OmniProcessor
    items = CFG['items']
    args.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    assert model_dir.is_dir(), model_dir
    print('MODEL_DIR', model_dir, '\nMODEL_COMMIT', REVISION, flush=True)
    prov = provenance()
    print('PROVENANCE', json.dumps(prov), flush=True)
    (args.out / 'config.json').write_text(json.dumps(dict(
        model_dir=str(model_dir), model_commit=REVISION, items='configs/c1_ext_devel100.json', n=len(items), conditions=CONDS,
        system=SYSTEM, prompt_P2=PROMPT_P2, prompt_T2=PROMPT_T2, t2_user_content='[text: reference sentence + newline, text: prompt_T2]',
        decoding=dict(do_sample=False, max_new_tokens=args.max_new_tokens, batch=1, talker='disabled', use_audio_in_video=False),
        dry_run=args.dry_run, **prov), indent=1, ensure_ascii=False) + '\n')
    if args.dry_run:
        t0 = time.perf_counter()
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        print('PROCESSOR_LOAD_S', round(time.perf_counter() - t0, 1), flush=True)
        for it in items[:args.dry_run]:
            for cond in CONDS:
                text, batch = build(processor, cond, it, args.audio_dir)
                fake = json.dumps(dict(transcript=it['sentence'], intent=it['intent'],
                                       entities=[dict(type=t, value=v) for t, v in it['entities']]))
                print('DRY', cond, it['file'], 'chars', len(text), 'shapes', json.dumps({k: list(v.shape) for k, v in batch.items()}),
                      'parse', parse(fake)[1], 'audio_tok' if '<|AUDIO|>' in text else 'no_audio_tok', flush=True)
                (args.out / f'dry_{cond}_{it["slurp_id"]}.txt').write_text(text)
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
        d = args.out / cond; d.mkdir()
        torch.cuda.reset_peak_memory_stats(); tot = 0.0; t_cond = time.perf_counter()
        with open(d / 'raw.jsonl', 'w') as fh:
            for it in items:
                text, batch = build(processor, cond, it, args.audio_dir)
                batch = batch.to(model.device)
                for k, v in batch.items():
                    if torch.is_floating_point(v):
                        batch[k] = v.to(model.dtype)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=args.max_new_tokens,
                                         do_sample=False, use_audio_in_video=False)
                torch.cuda.synchronize(); sec = time.perf_counter() - t1; tot += sec
                raw = processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                rec = dict(condition=cond, file=it['file'], slurp_id=it['slurp_id'], ref_sentence=it['sentence'], ref_intent=it['intent'],
                           ref_entities=it['entities'], prompt_text=text, raw_output=raw, parse_status=parse(raw)[1], seconds=sec,
                           input_tokens=int(batch['input_ids'].shape[1]), new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1]))
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
        summ = dict(condition=cond, n=len(items), infer_seconds_total=tot, wall_seconds=time.perf_counter() - t_cond,
                    infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **load)
        (d / 'summary.json').write_text(json.dumps(summ, indent=2) + '\n')
        print('COND_DONE', json.dumps(summ), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
