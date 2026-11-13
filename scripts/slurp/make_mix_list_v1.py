"""Draw the mix_spec_v1 devel list: 100 swap groups (docs/mix_spec_v1.md). LIST ONLY; no audio.
Pairs: devel candidate pairs (split_v2_seed2026) after the mix_spec_v1 filter (same gender, |dd| <= 1 s, D_T and D_I
  non-empty, no shared stem / containment between D_T and D_I values; scripts/slurp/devel_capacity_mixspec_v1.py).
Draw (seed 2026): kept pairs sorted by (file_a, file_b), random.Random(2026) shuffle, greedy accept while both recordings
  have been used < 3 times, until 100. Orientation (which recording is A) by the same rng.
RIRs: SMS-WSJ cv_dev93 only (never test). Angle strata on the separation |wrap(az1 - az0)| (results/sms_wsj_geometry,
  buckets [0,15) [15,30) [30,60) [60,180]): 25 groups per stratum; stratum labels shuffled over the groups; within a
  stratum, examples sorted by id and sampled without replacement. theta1 = az_rel_mic0 of RIR position 0 (h_0.wav),
  theta2 = position 1 (h_1.wav). Swap 1: A@theta1, B@theta2; swap 2: A@theta2, B@theta1; the question is always theta1.
Offsets (8 kHz samples, follow the speaker, identical in both swaps): the longer utterance starts at 0, the shorter at
  an integer offset drawn uniformly from [0, L_long - L_short]; L = ceil(n_16k / 2) (length after resample_poly 1/2).
Noise: per group, SNR ~ U(20, 30) dB and an integer noise seed, both from the same rng (used by the mixing script).
v2 mode (--v2; docs/mix_spec_v1.md "第二版"): after the same shuffle, the 100 pairs are chosen by a 0/1 ILP maximising
  sum (n_kept - rank) * x subject to: exactly 100 pairs, recording use <= 3, speaker groups <= 30, male groups >= 20.
  If infeasible: report max pairs under the constraints, and (100 pairs, use <= 3) max male groups and min max speaker
  groups; write nothing and exit 2. The chosen pairs are then processed in rank order with the same rng as v1.
Usage: python -I scripts/slurp/make_mix_list_v1.py DATA_DIR AUDIO_DIR OUT_JSON [--v2] [--split test --n 200]
--split test (2026-11-12, docs/slurp_test_freeze_v1.md §0-3/§4): split_v2 test pairs and SMS-WSJ test_eval92 geometry;
  same filter, ILP, rng sequence and constants (use <= 3, speaker groups <= 30, male groups >= 20). The ILP is tried with
  exactly n groups, then n-4, n-8, ... (four equal strata) down to 100; below 100 nothing is written (exit 2).
  The devel default (no --split) is unchanged. (Text after 'Usage' is not part of the stored rule string.)
"""
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from binding_score_v1 import qualified_pairs  # noqa: E402
from devel_capacity_mixspec_v1 import reason  # noqa: E402

SEED, N, K = 2026, 100, 3
MAX_SPK, MIN_MALE = 30, 20
BUCKETS = [(0, 15), (15, 30), (30, 60), (60, 180.0001)]
GEOM = PROJ / 'results/sms_wsj_geometry/cv_dev93_per_mixture.csv'
GEOM_SPLIT = dict(devel=GEOM, test=PROJ / 'results/sms_wsj_geometry/test_eval92_per_mixture.csv')


def bucket(sep):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= sep < hi:
            return i
    raise ValueError(sep)


def ilp_v2(order, rec, min_male=MIN_MALE, max_spk=MAX_SPK, n_exact=N, objective='rank'):
    """Rows: recordings (<= K), speakers (<= max_spk), male (>= min_male), total (== n_exact or free)."""
    m = len(order)
    files = sorted({f for p in order for f in p}); fi = {f: i for i, f in enumerate(files)}
    spk = sorted({rec[f]['usrid'] for f in files}); si = {u: i for i, u in enumerate(spk)}
    A = lil_matrix((len(files) + len(spk) + 2, m)); lo, hi = [], []
    for j, (a, b) in enumerate(order):
        A[fi[a], j] += 1; A[fi[b], j] += 1
        A[len(files) + si[rec[a]['usrid']], j] = 1; A[len(files) + si[rec[b]['usrid']], j] = 1
        A[len(files) + len(spk), j] = rec[a]['gender'] == 'M'
        A[len(files) + len(spk) + 1, j] = 1
    lo = [0] * len(files) + [0] * len(spk) + [min_male, n_exact if n_exact else 0]
    hi = [K] * len(files) + [max_spk] * len(spk) + [np.inf, n_exact if n_exact else np.inf]
    c = -(m - np.arange(m)) if objective == 'rank' else -np.ones(m)
    res = milp(c=c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(m), bounds=Bounds(0, 1),
               options=dict(time_limit=600))
    if res.status != 0:
        return None, res.message
    return [order[j] for j in np.flatnonzero(np.round(res.x) == 1)], res.message


def closest(order, rec):
    out = {}
    sel, msg = ilp_v2(order, rec, n_exact=None, objective='count'); out['max_pairs_all_constraints'] = len(sel) if sel else msg
    for mm in range(MIN_MALE, -1, -1):
        sel, _ = ilp_v2(order, rec, min_male=mm, max_spk=10**6)
        if sel:
            out['max_male_groups_with_100_and_use_le_3'] = mm; break
    for ms in range(MAX_SPK, 101):
        sel, _ = ilp_v2(order, rec, min_male=0, max_spk=ms)
        if sel:
            out['min_max_speaker_groups_with_100_and_use_le_3'] = ms; break
    return out


def main(data, audio_dir, out_json, v2=False, split='devel', n_target=N):
    global N
    rec, pairs = qualified_pairs(data, split)
    kept = sorted((a, b) for a, b, g, dd, Da, Db in pairs if g and dd <= 1.0 and Da and Db and not reason(Da, Db))
    rng = random.Random(SEED)
    order = kept[:]; rng.shuffle(order)
    use = collections.Counter(); pick = []
    if v2:
        if split == 'devel':
            sel, msg = ilp_v2(order, rec)
            print('ILP_V2', msg, flush=True)
            if sel is None:
                print('INFEASIBLE', json.dumps(closest(order, rec)), flush=True)
                sys.exit(2)
        else:
            tried = []
            for N in range(n_target, 99, -4):
                sel, msg = ilp_v2(order, rec, n_exact=N); tried.append(dict(n=N, feasible=sel is not None, msg=msg))
                print('ILP_V2', N, msg, flush=True)
                if sel is not None:
                    break
            if sel is None:
                print('INFEASIBLE_ABOVE_100', json.dumps(tried), flush=True)
                sys.exit(2)
        chosen = set(sel)
        for a, b in order:              # rank order; same rng sequence as v1 from here on
            if (a, b) in chosen:
                pick.append((a, b) if rng.random() < 0.5 else (b, a)); use[a] += 1; use[b] += 1
    else:
        for a, b in order:
            if use[a] < K and use[b] < K:
                pick.append((a, b) if rng.random() < 0.5 else (b, a)); use[a] += 1; use[b] += 1
                if len(pick) == N:
                    break
    assert len(pick) == N, len(pick)
    with open(GEOM_SPLIT[split], newline='') as fh:
        geo = sorted(csv.DictReader(fh), key=lambda r: r['id'])
    by_b = collections.defaultdict(list)
    for r in geo:
        by_b[bucket(float(r['separation']))].append(r)
    labels = [i for i in range(len(BUCKETS)) for _ in range(N // len(BUCKETS))]; rng.shuffle(labels)
    rirs = {i: rng.sample(by_b[i], labels.count(i)) for i in range(len(BUCKETS))}
    groups = []
    for g, ((a, b), lab) in enumerate(zip(pick, labels)):
        r = rirs[lab].pop()
        n16 = {f: sf.info(Path(audio_dir) / f).frames for f in (a, b)}
        L = {f: math.ceil(n / 2) for f, n in n16.items()}
        long_, short = (a, b) if L[a] >= L[b] else (b, a)
        off = {long_: 0, short: rng.randint(0, L[long_] - L[short])}
        groups.append(dict(
            group=g, A=dict(file=a, usrid=rec[a]['usrid'], gender=rec[a]['gender'], sentence=rec[a]['sentence'],
                            intent=rec[a]['intent'], entities=json.loads(rec[a]['values']), n_16k=n16[a], n_8k=L[a], offset_8k=off[a]),
            B=dict(file=b, usrid=rec[b]['usrid'], gender=rec[b]['gender'], sentence=rec[b]['sentence'],
                   intent=rec[b]['intent'], entities=json.loads(rec[b]['values']), n_16k=n16[b], n_8k=L[b], offset_8k=off[b]),
            T_8k=max(L[a] + off[a], L[b] + off[b]), abs_duration_diff_s=abs(float(rec[a]['seconds']) - float(rec[b]['seconds'])),
            rir=dict(example_id=r['id'], rir_dir=r['rir_dir'], t60=float(r['t60']), theta1=float(r['az_rel_mic0_0']),
                     theta2=float(r['az_rel_mic0_1']), separation=float(r['separation']),
                     stratum=f'[{BUCKETS[lab][0]},{min(BUCKETS[lab][1], 180):g}{")" if BUCKETS[lab][1] < 180 else "]"}'),
            swaps=[dict(swap=1, A_pos=0, B_pos=1, target='A', question_theta='theta1'),
                   dict(swap=2, A_pos=1, B_pos=0, target='B', question_theta='theta1')],
            noise=dict(snr_db=rng.uniform(20, 30), seed=rng.randrange(2**31))))
    c100 = {it['file'] for it in json.load(open(PROJ / 'configs/c1_ext_devel100.json'))['items']} if split == 'devel' else set()
    files = [x[k]['file'] for x in groups for k in 'AB']
    uc = collections.Counter(files)
    dd = [x['abs_duration_diff_s'] for x in groups]; t60 = [x['rir']['t60'] for x in groups]
    stats = dict(
        kept_pairs=len(kept), groups=len(groups), recordings=len(uc), recording_use_counts=dict(sorted(collections.Counter(uc.values()).items())),
        speakers=len({x[k]['usrid'] for x in groups for k in 'AB'}),
        speaker_group_counts=dict(collections.Counter(x[k]['usrid'] for x in groups for k in 'AB').most_common()),
        speakers_by_gender=dict(collections.Counter(rec[f]['gender'] for f in {rec[f]['usrid']: f for f in uc}.values())),
        mode='v2' if v2 else 'v1', constraints=dict(max_speaker_groups=MAX_SPK, min_male_groups=MIN_MALE) if v2 else None,
        gender_groups=dict(collections.Counter(x['A']['gender'] for x in groups)),
        intents=len({x['A']['intent'] for x in groups}), intent_groups=dict(collections.Counter(x['A']['intent'] for x in groups).most_common()),
        duration_diff_s=dict(min=min(dd), median=sorted(dd)[len(dd) // 2], max=max(dd),
                             hist={f'[{i/4:g},{(i+1)/4:g})': sum(i / 4 <= v < (i + 1) / 4 for v in dd) for i in range(4)}),
        rir_examples_unique=len({x['rir']['example_id'] for x in groups}),
        angle_strata=dict(collections.Counter(x['rir']['stratum'] for x in groups)),
        t60=dict(min=min(t60), median=sorted(t60)[len(t60) // 2], max=max(t60),
                 hist={f'[{0.2 + i*0.1:.1f},{0.3 + i*0.1:.1f})': sum(0.2 + i * 0.1 <= v < 0.3 + i * 0.1 + (i == 2) * 1e-9 for v in t60) for i in range(3)}),
        overlap_with_c1_ext_devel100=dict(recordings=len(set(uc) & c100),
                                          groups_with_any=sum(x['A']['file'] in c100 or x['B']['file'] in c100 for x in groups),
                                          groups_with_both=sum(x['A']['file'] in c100 and x['B']['file'] in c100 for x in groups)),
        source_split_pairs=dict(collections.Counter('+'.join(sorted((rec[x['A']['file']]['split'], rec[x['B']['file']]['split']))) for x in groups)))
    if split != 'devel':
        stats.update(split=split, rir_set='test_eval92', n_target=n_target, ilp_tried=tried, overlap_with_c1_ext_devel100=None)
    Path(out_json).write_text(json.dumps(dict(seed=SEED, spec='docs/mix_spec_v1.md', rule=__doc__.split('Usage')[0],
                                              stats=stats, groups=groups), indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    ex = sys.argv[4:]
    sp = ex[ex.index('--split') + 1] if '--split' in ex else 'devel'; assert sp in ('devel', 'test')
    nt = int(ex[ex.index('--n') + 1]) if '--n' in ex else N
    assert sp == 'test' or nt == N
    main(*sys.argv[1:4], v2='--v2' in ex, split=sp, n_target=nt)
