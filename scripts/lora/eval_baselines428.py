"""Descriptive evaluation of the step 1-2 baselines (docs/lora_v2_runs.md §10, §13). CPU, reads raw outputs only.
Conditions on 428 devel recordings: P2 (clean), T = Whisper + T2 (clean), P2-R / T-R (C1-R, cv_dev93 single-talker reverb),
and the DIAGNOSTIC text LoRA (1e-4 step 296) on the clean Whisper transcripts (T0LoRA_W) vs T (plan §6 item 5 thresholds:
any-gold-value-hit and intent not lower by more than 0.03, parse failures <= 2%; on the selection half, all also reported).
Metrics = eval_c0_devel428.summary (unchanged) + Whisper WER (wer_counts on normalize_v2, as eval_c7w_f1) +
consistency with the existing 124-recording runs (P2 C1 job 22014089, W_C1 / T_C1 job 22016221) + train Whisper coverage.
Usage: python -I -B scripts/lora/eval_baselines428.py OUT_DIR
"""
import collections
import glob
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/slurp')); sys.path.insert(0, str(PROJ / 'scripts/lora'))
from eval_c0_devel428 import summary  # noqa: E402
from eval_text_lora_gate import boot, hits  # noqa: E402
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import wer_counts  # noqa: E402

O = PROJ / 'outputs'
SETS = dict(P2='lora_base_audio/p2_devel_clean/*/P2', T='lora_base_audio/whisper_t2_devel_clean/*/T', W='lora_base_audio/whisper_t2_devel_clean/*/W',
            P2_R='lora_base_audio/p2_devel_rev/*/P2', T_R='lora_base_audio/whisper_t2_devel_rev/*/T', W_R='lora_base_audio/whisper_t2_devel_rev/*/W',
            T0LoRA_W='lora_text_asr/lr1e-4_step296/*/T0LoRA_W')
OLD = dict(P2=O / 'qwen_omni_c3_p2/22014089/C1/raw.jsonl', W=O / 'qwen_omni_c7w_f1/22016221/W_C1/raw.jsonl', T=O / 'qwen_omni_c7w_f1/22016221/T_C1/raw.jsonl')


def load(pat):
    R = {}
    for p in sorted(glob.glob(str(O / pat / 'raw.jsonl'))):
        for l in open(p):
            r = json.loads(l); assert r['file'] not in R; R[r['file']] = r
    assert len(R) == 428, (pat, len(R))
    return R


def wer(R):
    e = n = 0
    for r in R.values():
        a, b = wer_counts(normalize_v2(r['ref_sentence']), normalize_v2(r['raw_output'])); e += a; n += b
    return e / n


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    D = {k: load(v) for k, v in SETS.items()}
    files = set(D['P2']); assert all(set(v) == files for v in D.values())
    ids = sorted({r['slurp_id'] for r in D['P2'].values()}); random.Random(2026).shuffle(ids); sel = set(ids[:len(ids) // 2])
    H = dict(selection={f for f in files if D['P2'][f]['slurp_id'] in sel}, report={f for f in files if D['P2'][f]['slurp_id'] not in sel}, all=files)
    rep = dict(spec='docs/lora_v2_runs.md §10, §13', conditions={}, wer={})
    keys = ('any_gold_value_hit', 'hit', 'gold_entities', 'date_recall', 'classes')
    for k in ('P2', 'T', 'P2_R', 'T_R', 'T0LoRA_W'):
        rep['conditions'][k] = {}
        for h, fs in H.items():
            s = summary({f: D[k][f] for f in fs})
            rep['conditions'][k][h] = dict({m: s[m] for m in keys}, intent_strict_acc=s['slu_v2']['intent_strict_acc'], parse_failures=s['slu_v2']['parse_failures'],
                                           parse_failure_rate=s['slu_v2']['parse_failures'] / len(fs), n=len(fs))
    for k in ('W', 'W_R'):
        rep['wer'][k] = {h: wer({f: D[k][f] for f in fs}) for h, fs in H.items()}
    # T inputs must be exactly the W transcripts (same prompt); T0LoRA_W inputs must equal T inputs
    rep['input_identity'] = dict(T_vs_T0LoRA_W_prompt_text=sum(D['T'][f]['prompt_text'] == D['T0LoRA_W'][f]['prompt_text'] for f in files),
                                 T_asr_equals_W=sum(D['T'][f]['asr_text'] == D['W'][f]['raw_output'] for f in files),
                                 T_R_asr_equals_W_R=sum(D['T_R'][f]['asr_text'] == D['W_R'][f]['raw_output'] for f in files))
    # degradation check (plan §6 items 1 and 5) for the diagnostic, selection half primary
    hT, hL = hits(D['T']), hits(D['T0LoRA_W']); chk = {}
    for h, fs in H.items():
        a, b = rep['conditions']['T0LoRA_W'][h], rep['conditions']['T'][h]
        chk[h] = dict(any_delta=a['any_gold_value_hit'] - b['any_gold_value_hit'], intent_delta=a['intent_strict_acc'] - b['intent_strict_acc'],
                      parse_failure_rate=a['parse_failure_rate'], bootstrap_report_only=boot(fs, D['T'], hL, hT))
        chk[h]['passes_point_thresholds'] = chk[h]['any_delta'] >= -0.03 and chk[h]['intent_delta'] >= -0.03 and a['parse_failure_rate'] <= 0.02
    rep['diagnostic_text_lora_on_whisper'] = dict(note='DIAGNOSTIC; not a step-2 result', **chk)
    # consistency with existing 124-recording runs
    cons = {}
    for k, p in OLD.items():
        old = {json.loads(l)['file']: json.loads(l) for l in open(p)}; com = sorted(set(old) & files)
        cons[k] = dict(n=len(com), identical_raw_output=sum(old[f]['raw_output'] == D[k][f]['raw_output'] for f in com))
    rep['consistency_124'] = cons
    # train Whisper coverage
    TW = {}
    for p in sorted(glob.glob(str(O / 'lora_base_audio/whisper_train_all/*/W/raw.jsonl'))):
        for l in open(p):
            r = json.loads(l); k = (r['audio'], r['file']); assert k not in TW; TW[k] = r
    rep['train_whisper'] = dict(n=len(TW), clean=sum(a == 'clean' for a, _ in TW), rev=sum(a == 'rev' for a, _ in TW),
                                wer_clean=wer({f: r for (a, f), r in TW.items() if a == 'clean'}), wer_rev=wer({f: r for (a, f), r in TW.items() if a == 'rev'}),
                                empty_transcripts=sum(not r['raw_output'].strip() for r in TW.values()))
    (out / 'eval_baselines428.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    for k, v in rep['conditions'].items():
        print(k, ' | '.join(f"{h}: any {x['any_gold_value_hit']:.3f} date {x['date_recall']:.3f} int {x['intent_strict_acc']:.3f} pf {x['parse_failures']}" for h, x in v.items()))
        print('   classes all', v['all']['classes'])
    print('WER', json.dumps(rep['wer'])); print('IDENT', rep['input_identity']); print('CONS', cons)
    for h, x in chk.items():
        print('DIAG', h, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k != 'bootstrap_report_only'}), x['bootstrap_report_only']['ci95'])
    print('TRAIN_W', json.dumps(rep['train_whisper']))


if __name__ == '__main__':
    main(sys.argv[1])
