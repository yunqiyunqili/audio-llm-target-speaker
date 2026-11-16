English translation of c7_planA_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c7_planA_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C7 plan A v1: Qwen transcribes first, then T2 extracts (c7-planA-v1; 2026-11-05; written before the run)

- Input: the target stream of C5, exactly the same audio that C5 sends to Qwen: FaSNet-TAC v3 stream selection plus polarity correction → `resample_poly(x, 2, 1)[:2T]` → −30 dBFS. **Depends on the oracle azimuth**, as C5 does.
- First step, transcription: Qwen2.5-Omni-7B (commit ae9e1690, the same local snapshot), with the default SYSTEM as system prompt; the user content is [audio, text], where the text is:
  > Transcribe the speech in this audio exactly. Output only the transcription.
  - This sentence is exactly `PROMPT` in `scripts/qwen/qwen_omni_single.py`, used on clean speech in job 22003339. **Only this version is run; the prompt is not tuned.**
  - Decoding: greedy, max_new_tokens 256, batch 1, text output only.
- Second step, extraction: the raw transcript text of the first step, stripped of leading and trailing whitespace and followed by a newline, is used as text input, followed by the frozen **T2 prompt** (`qwen_omni_c1_p2t2.PROMPT_T2`), with no audio. The user content is [transcript + newline, PROMPT_T2], in the same format as T2. Decoding settings as above.
- Whisper is not run, and no model is downloaded.
- Evaluation:
  - With slu-score-v2 and binding-score-v1 (`eval_c4_v1.score` and `summ`), the same metrics as C5: the four class proportions, target-any, all target values hit, pair-flip accuracy, M, U, parse failures, refusals.
  - Also report the WER of the first-step transcript against the target sentence (normalize_v2), and against the interferer sentence.
  - Compared with C5 and C4-direct. Descriptive only, no verdict.
- Implementation: `scripts/qwen/qwen_omni_c6c7.py` (in the same job as the other stream of C6δ); evaluation `scripts/qwen/eval_c7_planA.py`.

## Results (2026-11-05; **depends on the oracle azimuth**, as C5 does)
- Run record: job 22015445 (the same job as C6δ); transcription C7ASR took 185.9 s, extraction C7T2 took 329.3 s. Evaluation: `scripts/qwen/eval_c7_planA.py`, results in `outputs/c7_planA_v1/eval_c7_planA.json`.

| Condition | Target only | Interferer only | Neither | target-any | All target values hit | Pair-flip accuracy | M | U | Parse failures / refusals |
|---|---|---|---|---|---|---|---|---|---|
| **C7 plan A** | 0.215 | 0.005 | 0.780 | **0.255** | 0.215 | 0.06 | 33 | 139 | **11 / 5** |
| C5 | 0.340 | 0.010 | 0.650 | 0.395 | 0.340 | 0.11 | 111 | 82 | 0 / 0 |
| C4-direct | 0.565 | 0 | 0.435 | 0.670 | 0.565 | 0.36 | 185 | 15 | 0 / 0 |

- Quality of the first-step transcripts:
  - WER against the target sentence 3.96, against the interferer sentence 4.28;
  - only 30/200 have WER < 0.5 against the target sentence;
  - 122/200 outputs are more than 2 times as long as the reference sentence;
  - 60/200 contain words such as "sorry", "can't", "I don't know";
  - 2 contain non-English characters.
- Examples:
  - the transcript of g001_s1 is "I don't know what your favorite song by Bruno Mars is. You should tell me so we can talk about it.";
  - the transcript of g001_s2 is "I'm sorry, I can't hear the audio. …".
- Description: with this version of the transcription prompt, Qwen on separated speech **often does not transcribe but responds directly to the content of the command**. C7 plan A is therefore 0.14 below C5. As agreed, only this version of the prompt was run, without adjustment. Whisper is not run, and no model is downloaded.
