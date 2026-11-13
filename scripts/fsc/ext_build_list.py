"""FSC extension v1 sample list (docs/fsc_extension_plan_v1.md §1). CPU; reads FSC valid metadata + audio headers only.

Single (1,240): FSC valid minus the 64 pilot recordings; texts sorted; rng = random.Random(2026); per text: eligible recordings
  sorted by path, rng.shuffle, take in order one per speaker until 5, then fill in the same order if needed.
  slurp_train_overlap = normalized transcription equals a SLURP train sentence (same norm as scripts/fsc/audit_and_freeze_valid.py).
  C1-R RIR (reverb_single.assignment rule): items sorted by item_id; Random(2026).sample(sorted test_eval92 ids, n); randrange(2).
Spatial (target 360 groups; stop if < 300): candidate pairs among the 1,240 = same gender, different speaker, |duration diff| <= 1 s,
  D_T and D_I non-empty on (slot=value) multisets (non-'none'); sorted by (path_a, path_b), rng2 = random.Random(2026) shuffle;
  0/1 ILP (make_mix_list_v1.ilp_v2, unchanged) on the first 20,000 ranked candidates: exactly N groups, recording use <= 3,
  speaker groups <= 25% of N, male groups >= 10% of N; N = 360, 356, ... >= 300. Then the mix_v1 rng sequence (rng2):
  orientation, stratum labels (N/4 each), test_eval92 examples per stratum sampled without replacement, offsets, noise.
Output: configs/fsc_ext_v1_list.json (groups in mix_v1 format, so scripts/sms_wsj/mix_slurp_v1.py mixes them unchanged).
Usage: python -I scripts/fsc/ext_build_list.py OUT_JSON
"""
import collections
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/slurp'))
from make_mix_list_v1 import BUCKETS, bucket, ilp_v2  # noqa: E402

FSC = ROOT / 'data/fsc/source/fluent_speech_commands_dataset'
RIRS = Path('<SEPARATION_DATA_ROOT>/data/sms_wsj/rirs')
SEED, PER_TEXT, N_TARGET, N_MIN, K_CAND = 2026, 5, 360, 300, 20000
SLOTS = ('action', 'object', 'location')
norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s.lower())).strip()  # noqa: E731  (audit_and_freeze_valid.py)


def sv(t):
    return collections.Counter(f'{s}={t[s]}' for s in SLOTS if t[s] != 'none')


def main(out_json):
    V = list(csv.DictReader(open(FSC / 'data/valid_data.csv')))
    gender = {r['speakerId']: r['gender'] for r in csv.DictReader(open(FSC / 'data/speaker_demographics.csv'))}
    used = {json.loads(x)['path'] for x in open(ROOT / 'outputs/fsc_pilot/valid64_manifest.jsonl')}
    slurp = {norm(json.loads(x)['sentence']) for x in open(ROOT / 'data/slurp/annotations/8eb1654/train.jsonl')}
    el = [r for r in V if r['path'] not in used]; assert len(V) == 3118 and len(el) == 3054
    by = collections.defaultdict(list)
    for r in el:
        by[r['transcription']].append(r)
    rng = random.Random(SEED); single = []
    for text in sorted(by):
        rs = sorted(by[text], key=lambda r: r['path']); rng.shuffle(rs)
        pick, spk = [], set()
        for r in rs:
            if len(pick) < PER_TEXT and r['speakerId'] not in spk:
                pick.append(r); spk.add(r['speakerId'])
        for r in rs:
            if len(pick) < PER_TEXT and r not in pick:
                pick.append(r)
        assert len(pick) == PER_TEXT
        single += pick
    assert len(single) == 248 * PER_TEXT
    items = []
    for r in sorted(single, key=lambda r: r['path']):
        n16 = sf.info(FSC / r['path']).frames
        items.append(dict(item_id=Path(r['path']).stem, path=r['path'], text=r['transcription'], speaker=r['speakerId'], gender=gender[r['speakerId']],
                          target={s: r[s] for s in SLOTS}, slurp_train_overlap=norm(r['transcription']) in slurp, n_16k=n16))
    assert len({it['item_id'] for it in items}) == len(items)
    sc = json.load(open(RIRS / 'scenarios.json'))['datasets']['test_eval92']
    rr = random.Random(SEED); ex = rr.sample(sorted(sc, key=int), len(items))
    for it, e in zip(sorted(items, key=lambda x: x['item_id']), ex):
        it['reverb'] = dict(rir_set='test_eval92', rir_example=e, rir_pos=rr.randrange(2))
    # ---------------- pairs
    rec = {it['path']: dict(usrid=it['speaker'], gender='M' if it['gender'] == 'male' else 'F') for it in items}
    P = {it['path']: it for it in items}
    cand = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a['gender'] == b['gender'] and a['speaker'] != b['speaker'] and abs(a['n_16k'] - b['n_16k']) / 16000 <= 1.0:
                if (sv(a['target']) - sv(b['target'])) and (sv(b['target']) - sv(a['target'])):
                    cand.append(tuple(sorted((a['path'], b['path']))))
    cand = sorted(set(cand)); rng2 = random.Random(SEED); order = cand[:]; rng2.shuffle(order); order = order[:K_CAND]
    tried, sel = [], None
    for N in range(N_TARGET, N_MIN - 1, -4):
        sel, msg = ilp_v2(order, rec, min_male=math.ceil(0.10 * N), max_spk=int(0.25 * N), n_exact=N); tried.append(dict(n=N, feasible=sel is not None, msg=msg))
        print('ILP', N, msg, flush=True)
        if sel is not None:
            break
    if sel is None:
        print('STOP_FEWER_THAN_300_GROUPS', json.dumps(tried)); sys.exit(2)
    chosen = set(sel); pick = []
    for a, b in order:
        if (a, b) in chosen:
            pick.append((a, b) if rng2.random() < 0.5 else (b, a))
    with open(ROOT / 'results/sms_wsj_geometry/test_eval92_per_mixture.csv', newline='') as fh:
        geo = sorted(csv.DictReader(fh), key=lambda r: r['id'])
    byb = collections.defaultdict(list)
    for r in geo:
        byb[bucket(float(r['separation']))].append(r)
    labels = [i for i in range(len(BUCKETS)) for _ in range(N // len(BUCKETS))]; rng2.shuffle(labels)
    rirs = {i: rng2.sample(byb[i], labels.count(i)) for i in range(len(BUCKETS))}
    groups = []
    for g, ((a, b), lab) in enumerate(zip(pick, labels)):
        r = rirs[lab].pop(); A, B = P[a], P[b]
        L8 = {f: math.ceil(P[f]['n_16k'] / 2) for f in (a, b)}
        long_, short = (a, b) if L8[a] >= L8[b] else (b, a)
        off = {long_: 0, short: rng2.randint(0, L8[long_] - L8[short])}
        ent = lambda it: [[s, it['target'][s]] for s in SLOTS if it['target'][s] != 'none']  # noqa: E731
        side = lambda it, f: dict(file=f, item_id=it['item_id'], usrid=it['speaker'], gender=rec[f]['gender'], sentence=it['text'], intent='fsc', entities=ent(it),  # noqa: E731
                                  target=it['target'], n_16k=it['n_16k'], n_8k=L8[f], offset_8k=off[f])
        groups.append(dict(group=g, A=side(A, a), B=side(B, b), T_8k=max(L8[a] + off[a], L8[b] + off[b]), abs_duration_diff_s=abs(A['n_16k'] - B['n_16k']) / 16000,
                           rir=dict(example_id=r['id'], rir_dir=r['rir_dir'], t60=float(r['t60']), theta1=float(r['az_rel_mic0_0']), theta2=float(r['az_rel_mic0_1']),
                                    separation=float(r['separation']), stratum=f'[{BUCKETS[lab][0]},{min(BUCKETS[lab][1], 180):g}{")" if BUCKETS[lab][1] < 180 else "]"}'),
                           swaps=[dict(swap=1, A_pos=0, B_pos=1, target='A', question_theta='theta1'), dict(swap=2, A_pos=1, B_pos=0, target='B', question_theta='theta1')],
                           noise=dict(snr_db=rng2.uniform(20, 30), seed=rng2.randrange(2**31))))
    use = collections.Counter(x[k]['file'] for x in groups for k in 'AB')
    stats = dict(single=len(items), texts=len({it['text'] for it in items}), speakers=len({it['speaker'] for it in items}),
                 single_gender=dict(collections.Counter(it['gender'] for it in items)), slurp_train_overlap_items=sum(it['slurp_train_overlap'] for it in items),
                 slurp_train_overlap_texts=len({it['text'] for it in items if it['slurp_train_overlap']}), candidate_pairs=len(cand), ilp_candidates=len(order),
                 ilp_tried=tried, groups=len(groups), samples=2 * len(groups), recordings_in_groups=len(use), recording_use_counts=dict(collections.Counter(use.values())),
                 speaker_group_counts=dict(collections.Counter(x[k]['usrid'] for x in groups for k in 'AB').most_common()),
                 gender_groups=dict(collections.Counter(x['A']['gender'] for x in groups)), angle_strata=dict(collections.Counter(x['rir']['stratum'] for x in groups)),
                 rir_examples_unique=len({x['rir']['example_id'] for x in groups}), reverb_examples_unique=len({it['reverb']['rir_example'] for it in items}))
    Path(out_json).write_text(json.dumps(dict(seed=SEED, spec='docs/fsc_extension_plan_v1.md', rule=__doc__.split('Usage')[0], stats=stats, single=items, groups=groups),
                                         indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
