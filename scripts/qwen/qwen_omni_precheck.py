"""CPU pre-check before resubmitting qwen_omni_single (no model weights loaded).
Loads Qwen2_5OmniProcessor from the local snapshot dir and runs the same chat template + audio
preprocessing as scripts/qwen/qwen_omni_single.py on the two wavs already written by job 21999584
(those wavs were deleted on 2026-11-05 as sample-identical duplicates of outputs/qwen_omni_single/22003339/).
Usage: python -I scripts/qwen/qwen_omni_precheck.py WAV_DIR
"""
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_omni_single import PROMPT, REVISION, SR, SYSTEM  # noqa: E402


def main(wav_dir):
    from transformers import Qwen2_5OmniProcessor
    model_dir = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen2.5-Omni-7B/snapshots' / REVISION
    print('MODEL_DIR', model_dir, flush=True)
    t0 = time.perf_counter()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
    print(f'PROCESSOR_OK {time.perf_counter() - t0:.1f}s', type(processor.feature_extractor).__name__,
          'fe_sr', processor.feature_extractor.sampling_rate, flush=True)
    conversation = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                    {'role': 'user', 'content': [{'type': 'audio', 'audio': 'placeholder'},
                                                 {'type': 'text', 'text': PROMPT}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    for name in ('C1_original', 'C1b_8k_up16k'):
        wav, sr = sf.read(Path(wav_dir) / f'{name}.wav', dtype='float32')
        assert sr == SR and wav.ndim == 1, (sr, wav.shape)
        batch = processor(text=text, audio=[wav], sampling_rate=SR, return_tensors='pt', padding=True)
        info = {k: [list(v.shape), str(v.dtype)] for k, v in batch.items()}
        print('INPUT_OK', name, f'{len(wav) / SR:.3f}s', json.dumps(info), flush=True)
    print('PRECHECK_PASS')


if __name__ == '__main__':
    main(sys.argv[1])
