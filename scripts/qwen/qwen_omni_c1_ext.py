"""C1 extension: Qwen2.5-Omni-7B structured SLU on 100 DEVEL recordings (configs/c1_ext_devel100.json), 3 conditions:
  P0 = audio + current prompt (identical to scripts/qwen/qwen_omni_c1_semantic.py PROMPT)
  P1 = audio + current prompt + 3 fixed text examples from v2 TRAIN (configs/c1_ext_devel100.json: examples_train)
  T0 = reference transcript (SLURP `sentence`) as a text item + current prompt, NO audio
Same items, same decoding for all (greedy, max_new_tokens 256, batch 1, text only / talker disabled).
Writes OUT/config.json (prompts, examples, decoding) and OUT/{P0,P1,T0}/raw.jsonl (full chat-template text per item,
raw output, timing, token counts) + OUT/{cond}/summary.json (time, peak GPU). Scoring is offline
(scripts/qwen/rescore_slu_v2.py); nothing is scored here.
--dry-run: CPU only, NO weights: loads the processor, builds all 3 conditions for the first N items, runs the
processor, and checks the parser/scorer on a synthetic reference-equal output.
Usage: python scripts/qwen/qwen_omni_c1_ext.py --audio-dir DIR --out DIR [--dry-run N]
"""
import argparse
import json
import os
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
CONDS = ('P0', 'P1', 'T0')


def ref_json(sentence, intent, entities):
    return json.dumps(dict(transcript=sentence, intent=intent, entities=[dict(type=t, value=v) for t, v in entities]))


PROMPT_P1 = PROMPT + '\nExamples (for the output format):\n' + '\n'.join(
    f'Command: "{e["sentence"]}"\nOutput: {ref_json(e["sentence"], e["intent"], e["entities"])}' for e in CFG['examples_train'])


def conversation(cond, it):
    if cond == 'T0':
        user = [{'type': 'text', 'text': it['sentence'] + '\n'}, {'type': 'text', 'text': PROMPT}]
    else:
        user = [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': PROMPT_P1 if cond == 'P1' else PROMPT}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': user}]


def build(processor, cond, it, audio_dir):
    text = processor.apply_chat_template(conversation(cond, it), add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):          # transformers 4.52.4 returns a 1-element list here
        assert len(text) == 1, len(text)
        text = text[0]
    if cond == 'T0':
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
    (args.out / 'config.json').write_text(json.dumps(dict(
        model_dir=str(model_dir), commit=REVISION, items='configs/c1_ext_devel100.json', n=len(items), conditions=CONDS,
        system=SYSTEM, prompt_P0_T0=PROMPT, prompt_P1=PROMPT_P1, examples_train=CFG['examples_train'],
        t0_user_content='[text: reference sentence + newline, text: prompt]', decoding=dict(do_sample=False, max_new_tokens=args.max_new_tokens,
        batch=1, talker='disabled', use_audio_in_video=False), dry_run=args.dry_run), indent=1, ensure_ascii=False) + '\n')
    if args.dry_run:
        from rescore_slu_v2 import score_item
        t0 = time.perf_counter()
        processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        print('PROCESSOR_LOAD_S', round(time.perf_counter() - t0, 1), flush=True)
        for it in items[:args.dry_run]:
            for cond in CONDS:
                text, batch = build(processor, cond, it, args.audio_dir)
                shapes = {k: list(v.shape) for k, v in batch.items()}
                fake = ref_json(it['sentence'], it['intent'], it['entities'])
                obj, st = parse(fake)
                s = score_item(dict(raw_output=fake, ref_entities=it['entities'], ref_intent=it['intent'], ref_sentence=it['sentence']))
                print('DRY', cond, it['file'], 'chars', len(text), 'shapes', json.dumps(shapes), 'parse', st,
                      'full_v2', s['full_v2'], 'audio_tok' if '<|AUDIO|>' in text else 'no_audio_tok', flush=True)
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
