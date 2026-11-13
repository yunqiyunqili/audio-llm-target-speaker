"""Qwen2.5-Omni-7B C1 (clean close-talk, original 16 kHz) structured SLU pilot on 20 DEVEL recordings.
NOT SUBMITTED (plan: docs/c1_semantic_plan.md). Text output only, greedy decoding, batch size 1.
Items: configs/c1_semantic_devel20.json. Labels in prompt: configs/slurp_label_inventory_train.json.
Loads the model from the local snapshot dir (commit ae9e1690), offline, exactly as scripts/qwen/qwen_omni_single.py.
Writes OUT/raw.jsonl (raw text + parse + per-item scores) and OUT/summary.json.
Usage: python scripts/qwen/qwen_omni_c1_semantic.py --audio-dir DIR --out DIR
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import REVISION, SR, SYSTEM  # noqa: E402
from slu_normalize import entity_counts, normalize, parse, wer_counts  # noqa: E402

INV = json.load(open(PROJ / 'configs/slurp_label_inventory_train.json'))
PROMPT = ('This audio is a spoken command to a voice assistant. Return ONLY one JSON object, with no other text, '
          'in exactly this format:\n'
          '{"transcript": "<the words spoken>", "intent": "<one intent label>", '
          '"entities": [{"type": "<one entity type>", "value": "<the entity words as spoken>"}]}\n'
          'Use "entities": [] if there are none.\n'
          'Allowed intent labels: ' + ', '.join(INV['intents']) + '\n'
          'Allowed entity types: ' + ', '.join(INV['entity_types']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    args = ap.parse_args()
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    items = json.load(open(PROJ / 'configs/c1_semantic_devel20.json'))['items']
    args.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    assert model_dir.is_dir(), model_dir
    print('MODEL_DIR', model_dir, '\nMODEL_COMMIT', REVISION, flush=True)
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
    torch.cuda.synchronize(); load_s = time.perf_counter() - t0; load_peak = torch.cuda.max_memory_allocated() / 2**30
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'}, {'type': 'text', 'text': PROMPT}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    tot = dict(n=0, intent_correct=0, intent_invalid_label=0, parse=dict(), wer_err=0, wer_ref=0,
               ent_tp=0, ent_hyp=0, ent_ref=0, ent_tp_nospace=0, seconds=0.0)
    torch.cuda.reset_peak_memory_stats()
    with open(args.out / 'raw.jsonl', 'w') as fh:
        for it in items:
            wav, sr = sf.read(args.audio_dir / it['file'], dtype='float32')
            assert sr == SR and wav.ndim == 1, (it['file'], sr, wav.shape)
            batch = processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True).to(model.device)
            for k, v in batch.items():
                if torch.is_floating_point(v):
                    batch[k] = v.to(model.dtype)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=args.max_new_tokens,
                                     do_sample=False, use_audio_in_video=False)
            torch.cuda.synchronize(); sec = time.perf_counter() - t1
            raw = processor.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            obj, status = parse(raw)
            hyp = obj or dict(transcript='', intent='', entities=[], dropped_entities=0)
            e, n = wer_counts(normalize(it['sentence']), normalize(hyp['transcript']))
            tp, nh, nr = entity_counts(it['entities'], hyp['entities'])
            tp2, _, _ = entity_counts(it['entities'], hyp['entities'], nospace=True)
            rec = dict(file=it['file'], slurp_id=it['slurp_id'], ref_sentence=it['sentence'], ref_intent=it['intent'],
                       ref_entities=it['entities'], raw_output=raw, parse_status=status, hyp=hyp,
                       intent_correct=hyp['intent'] == it['intent'], intent_in_inventory=hyp['intent'] in INV['intents'],
                       wer_err=e, wer_ref=n, ent_tp=tp, ent_hyp=nh, ent_ref=nr, ent_tp_nospace=tp2, seconds=sec,
                       new_tokens=int(ids.shape[1] - batch['input_ids'].shape[1]))
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
            print('ITEM', json.dumps({k: rec[k] for k in ('file', 'parse_status', 'intent_correct', 'ent_tp', 'ent_ref', 'seconds')}), flush=True)
            tot['n'] += 1; tot['intent_correct'] += rec['intent_correct']; tot['intent_invalid_label'] += not rec['intent_in_inventory']
            tot['parse'][status] = tot['parse'].get(status, 0) + 1
            for k in ('wer_err', 'wer_ref', 'ent_tp', 'ent_hyp', 'ent_ref', 'ent_tp_nospace', 'seconds'):
                tot[k] += rec[k]
    f1 = lambda tp, h, r: 2 * tp / (h + r) if h + r else 0.0  # noqa: E731
    summary = dict(model_dir=str(model_dir), commit=REVISION, prompt=PROMPT, load_seconds=load_s, load_peak_gpu_gib=load_peak,
                   infer_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, **tot,
                   intent_accuracy=tot['intent_correct'] / tot['n'], wer=tot['wer_err'] / max(tot['wer_ref'], 1),
                   entity_precision=tot['ent_tp'] / max(tot['ent_hyp'], 1), entity_recall=tot['ent_tp'] / max(tot['ent_ref'], 1),
                   entity_f1=f1(tot['ent_tp'], tot['ent_hyp'], tot['ent_ref']),
                   entity_f1_nospace=f1(tot['ent_tp_nospace'], tot['ent_hyp'], tot['ent_ref']))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'prompt'}), flush=True)


if __name__ == '__main__':
    main()
