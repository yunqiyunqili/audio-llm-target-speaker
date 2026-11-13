"""Qwen2.5-Omni-7B single-utterance, text-only inference on one SLURP close-talk recording,
in two versions: original bandwidth (C1) and 16k->8k->16k (C1b). Records load time,
per-input inference time, peak GPU memory and generated text.

Requires envs/qwen_omni and the pinned snapshot Qwen/Qwen2.5-Omni-7B@ae9e1690 under $HF_HOME.
Fix after job 21999394 (approved option a): the model and processor are loaded from the LOCAL
snapshot directory $HF_HOME/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/<REVISION>, because
transformers 4.52.4 looks up spk_dict.pt with revision=None ("main"), and the cache has no
refs/main. The commit is thus fixed by the directory; offline mode is kept.

Usage: python scripts/qwen/qwen_omni_single.py --audio <file.flac> --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'
REVISION = 'ae9e1690543ffd5c0221dc27f79834d0294cba00'
SR = 16000
PROMPT = 'Transcribe the speech in this audio exactly. Output only the transcription.'
SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
          'capable of perceiving auditory and visual inputs, as well as generating text and speech.')


def c1b(x):
    """16 kHz -> 8 kHz -> 16 kHz, scipy polyphase (Kaiser window, default beta=5.0)."""
    return resample_poly(resample_poly(x, 1, 2), 2, 1)[:len(x)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-new-tokens', type=int, default=128)
    args = ap.parse_args()
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    import transformers

    x, sr = sf.read(args.audio, dtype='float32', always_2d=False)
    assert sr == SR and x.ndim == 1, (sr, x.shape)
    inputs = {'C1_original': x, 'C1b_8k_up16k': c1b(x)}
    args.out.mkdir(parents=True, exist_ok=False)
    for name, wav in inputs.items():
        sf.write(args.out / f'{name}.wav', wav, SR, subtype='FLOAT')

    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    assert model_dir.is_dir() and (model_dir / 'spk_dict.pt').exists(), model_dir
    print(f'MODEL_DIR {model_dir}', flush=True)
    print(f'MODEL_COMMIT {REVISION}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa', local_files_only=True)
    model.disable_talker()                       # text output only
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    load_peak = torch.cuda.max_memory_allocated() / 2**30

    conversation = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                    {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'},
                                                 {'type': 'text', 'text': PROMPT}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    results = []
    for name, wav in inputs.items():
        batch = processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)
        batch = batch.to(model.device)
        for k, v in batch.items():
            if torch.is_floating_point(v):
                batch[k] = v.to(model.dtype)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**batch, return_audio=False, thinker_max_new_tokens=args.max_new_tokens,
                                 do_sample=False, use_audio_in_video=False)
        torch.cuda.synchronize()
        new = ids[:, batch['input_ids'].shape[1]:]
        out = processor.batch_decode(new, skip_special_tokens=True)[0]
        results.append(dict(input=name, seconds=time.perf_counter() - t1, new_tokens=int(new.shape[1]),
                            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, output=out))
        print('RESULT ' + json.dumps(results[-1]), flush=True)
    report = dict(model=MODEL_ID, revision=REVISION, model_dir=str(model_dir), transformers=transformers.__version__, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), audio=str(args.audio), audio_seconds=len(x) / SR,
                  prompt=PROMPT, decoding=dict(do_sample=False, max_new_tokens=args.max_new_tokens),
                  c1b_resampler='scipy.signal.resample_poly 1/2 then 2/1', load_seconds=load_s,
                  load_peak_gpu_gib=load_peak, results=results)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
