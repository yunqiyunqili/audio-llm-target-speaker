"""Comparison of three-way speaker-disjoint SLURP splits (rule A) for test hold-out fractions 0.2/0.3/0.4.
Counts only: no audio, no manifests are written (formal lists come after a fraction is chosen).

Population and pairing: identical to scripts/slurp/make_slurp_speaker_heldout.py (SLURP commit 8eb1654,
headset recordings, conflicting file excluded, same pairing rule).
Speaker assignment (seed 2026, gender-stratified over known-gender F/M speakers):
  per gender: sort usrids, random.Random(2026).shuffle -> order (same order for every fraction)
  test  = order[:round(f * n)]
  devel = next round(0.2 * (n - n_test)) speakers of the same order
  train = the rest; UNK-gender speakers always go to train (never in devel/test, never same-gender pairs)
Sentences: official slurp_id split; split s keeps official-s sentences spoken by split-s speakers.
Cross-set identical sentence text (lower/strip) is then removed, keeping the evaluation side:
  text also in test  -> removed from train and devel;  text also in devel -> removed from train.
Reuse limit: maximum number of test pairs (same gender, |dur diff| <= 1 s) such that each recording
  is used at most K=3 times; exact via 0/1 ILP (scipy.optimize.milp / HiGHS), greedy given as a check.
Also: the existing S lists (heldout_seed2026) with cross-set identical texts removed, same rule,
  written as NEW files S_dedup_* next to the originals (originals untouched).
Usage: python -I scripts/slurp/compare_slurp_speaker_splits.py ANNOT_DIR DURATIONS_CSV HELDOUT_DIR
"""
import collections
import csv
import gzip
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_slurp_speaker_heldout import SEED, SPLITS, load, gender_of, pairs_in_split  # noqa: E402

TEST_FRACTIONS = (0.2, 0.3, 0.4)
DEVEL_FRACTION = 0.2
K = 3
RULE_A = lambda r: r['status'] == 'correct' and r['wer'] == 0 and r['ent_wer'] == 0  # noqa: E731
norm = lambda s: s.lower().strip()  # noqa: E731


def assign(recs, f_test):
    spk = sorted({(r['gender'], r['usrid']) for r in recs})
    out = {s: set() for s in SPLITS}; names = {}
    for g in ('F', 'M'):
        order = sorted(u for gg, u in spk if gg == g)
        random.Random(SEED).shuffle(order)
        kt = round(f_test * len(order)); kd = round(DEVEL_FRACTION * (len(order) - kt))
        out['test'] |= set(order[:kt]); out['devel'] |= set(order[kt:kt + kd]); out['train'] |= set(order[kt + kd:])
        names[g] = dict(test=sorted(order[:kt]), devel=sorted(order[kt:kt + kd]), n_pool=len(order))
    out['train'] |= {u for g, u in spk if g == 'UNK'}
    return out, names


def dedup(keep):
    """Remove cross-set identical texts keeping the evaluation side. Returns new dict + removal counts."""
    t_test = {norm(r['sentence']) for r in keep['test']}
    removed = {}
    new = dict(test=keep['test'])
    new['devel'] = [r for r in keep['devel'] if norm(r['sentence']) not in t_test]
    t_dev2 = {norm(r['sentence']) for r in new['devel']}
    new['train'] = [r for r in keep['train'] if norm(r['sentence']) not in t_test and norm(r['sentence']) not in t_dev2]
    for s in SPLITS:
        ids = {id(r) for r in new[s]}
        gone = [r for r in keep[s] if id(r) not in ids]
        removed[s] = dict(recordings=len(gone), texts=len({norm(r['sentence']) for r in gone}))
    return new, removed


def cross(keep):
    sid = {s: {r['slurp_id'] for r in keep[s]} for s in SPLITS}
    txt = {s: {norm(r['sentence']) for r in keep[s]} for s in SPLITS}
    spk = {s: {r['usrid'] for r in keep[s]} for s in SPLITS}
    return {f'{a}&{b}': dict(slurp_id=len(sid[a] & sid[b]), sentence_text=len(txt[a] & txt[b]),
                             speakers=len(spk[a] & spk[b])) for a, b in combinations(SPLITS, 2)}


def reuse_limited(sel, rows, k=K, lim=1.0):
    edges = [(i, j) for i, j, g, d in rows if g and d <= lim]
    if not edges:
        return dict(edges=0, ilp_max_pairs=0, greedy_pairs=0)
    e = np.array(edges); m = len(e); n = len(sel)
    A = coo_matrix((np.ones(2 * m), (np.r_[e[:, 0], e[:, 1]], np.r_[np.arange(m), np.arange(m)])), shape=(n, m))
    res = milp(c=-np.ones(m), constraints=LinearConstraint(A.tocsr(), 0, k), integrality=np.ones(m),
               bounds=Bounds(0, 1), options=dict(time_limit=600))
    x = np.round(res.x).astype(int) if res.x is not None else np.zeros(m, int)
    use = np.bincount(e[x == 1].ravel(), minlength=n)
    assert use.max() <= k
    deg = collections.Counter(e.ravel().tolist())
    used = collections.Counter(); greedy = 0
    for i, j in sorted(edges, key=lambda ij: deg[ij[0]] + deg[ij[1]]):
        if used[i] < k and used[j] < k:
            used[i] += 1; used[j] += 1; greedy += 1
    return dict(edges=m, ilp_status=res.message, ilp_optimal=bool(res.status == 0), ilp_max_pairs=int(x.sum()),
                ilp_recordings_used=int((use > 0).sum()),
                upper_bound_sum_min_deg_k_over_2=int(sum(min(v, k) for v in deg.values()) // 2), greedy_pairs=greedy)


def describe(sel):
    counts, rows = pairs_in_split(sel)
    spk = {r['usrid'] for r in sel}
    return dict(recordings=len(sel), speakers=len(spk),
                speakers_by_gender=dict(sorted(collections.Counter(gender_of(u) for u in spk).items())),
                sentences=len({r['slurp_id'] for r in sel}), sentence_texts=len({norm(r['sentence']) for r in sel}),
                **{k: counts.get(k, 0) for k in ('pairs', 'pairs_same_gender', 'pairs_same_gender_dur_le_1s',
                                                 'pairs_same_gender_dur_le_2s', 'recordings_with_partner')}), rows


def main(annot, durations, heldout_dir):
    recs = [r for r in load(annot, durations)]
    report = dict(seed=SEED, devel_fraction=DEVEL_FRACTION, reuse_limit=K, rule='A: status==correct & wer==0 & ent_wer==0',
                  annotations=annot, fractions={})
    for f in TEST_FRACTIONS:
        spk_sets, names = assign(recs, f)
        keep = {s: [r for r in recs if r['split'] == s and r['usrid'] in spk_sets[s] and RULE_A(r)] for s in SPLITS}
        before = cross(keep)
        keep, removed = dedup(keep)
        entry = dict(speakers_assigned={s: len(v) for s, v in spk_sets.items()}, heldout_names=names,
                     cross_before_text_dedup=before, removed_by_text_dedup=removed, cross_after=cross(keep), splits={})
        for s in SPLITS:
            d, rows = describe(keep[s])
            if s == 'test':
                d['reuse_limited_same_gender_dur_le_1s'] = reuse_limited(keep[s], rows)
            entry['splits'][s] = d
        report['fractions'][str(f)] = entry
        print(f, json.dumps(entry['splits']), json.dumps(entry['cross_after']), flush=True)

    # sensitivity lists (existing seed2026 S version): remove cross-set identical texts -> new files
    hd = Path(heldout_dir); S = {}
    for s in SPLITS:
        with open(hd / f'S_{s}_recordings.csv') as fh:
            S[s] = list(csv.DictReader(fh))
        for r in S[s]:
            r['seconds'] = float(r['seconds'])
    sens = dict(cross_before=cross(S))
    S2, removed = dedup(S)
    sens.update(removed=removed, cross_after=cross(S2), splits={})
    for s in SPLITS:
        d, rows = describe(S2[s])
        d['recordings_without_entities_not_pairable'] = sum(not r['types'] for r in S2[s])
        _, all_rows = pairs_in_split(S2[s])
        dd = [x[3] for x in all_rows]
        d['pairs_dur_le_1s'] = sum(x <= 1 for x in dd); d['pairs_dur_le_2s'] = sum(x <= 2 for x in dd)
        sens['splits'][s] = d
        with open(hd / f'S_dedup_{s}_recordings.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(S2[s][0])); w.writeheader(); w.writerows(S2[s])
        with gzip.open(hd / f'S_dedup_{s}_pairs.csv.gz', 'wt', newline='') as fh:
            w = csv.writer(fh); w.writerow(['file_a', 'file_b', 'same_gender', 'abs_duration_diff_s'])
            for i, j, g, dv in rows:
                w.writerow([S2[s][i]['file'], S2[s][j]['file'], int(g), dv])
    report['sensitivity_S_seed2026_text_dedup'] = sens
    print('S_dedup', json.dumps(sens), flush=True)
    (PROJ / 'results/slurp/split_comparison.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
