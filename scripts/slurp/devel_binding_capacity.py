"""Devel pair capacity under the binding filter (split_v2_seed2026). Statistics only; no audio, no final draw.
Filter: same gender, |duration diff| <= 1 s, each recording used at most 3 times, D_T and D_I both non-empty
(scripts/slurp/binding_score_v1.py: multisets of normalize_v2 values).
Max number of pairs: same 0/1 ILP as v2 (scripts/slurp/compare_slurp_speaker_splits.py: reuse_limited, scipy milp/HiGHS),
called unchanged; a copy of the same formulation is solved again here only to obtain the selected pairs for the
distributions (asserted equal size). Greedy = reuse_limited's greedy (edges sorted by endpoint degree sum): LOWER BOUND.
Also re-runs the ILP without the D filter (should reproduce the 197 pairs of split_v2_stats.json).
Usage: python -I scripts/slurp/devel_binding_capacity.py DATA_DIR OUT_DIR
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binding_score_v1 import qualified_pairs  # noqa: E402
from compare_slurp_speaker_splits import reuse_limited  # noqa: E402

K = 3


def ilp_select(n, edges, k=K):
    e = np.array(edges); m = len(e)
    A = coo_matrix((np.ones(2 * m), (np.r_[e[:, 0], e[:, 1]], np.r_[np.arange(m), np.arange(m)])), shape=(n, m))
    res = milp(c=-np.ones(m), constraints=LinearConstraint(A.tocsr(), 0, k), integrality=np.ones(m),
               bounds=Bounds(0, 1), options=dict(time_limit=600))
    return [edges[i] for i in np.flatnonzero(np.round(res.x) == 1)], res.status == 0


def describe(sel_pairs, files, rec):
    used = sorted({files[i] for p in sel_pairs for i in p})
    spk = {rec[f]['usrid'] for f in used}
    return dict(pairs=len(sel_pairs), recordings=len(used), speakers=len(spk),
                speakers_by_gender=dict(collections.Counter(rec[f]['gender'] for f in {rec[x]['usrid']: x for x in used}.values())),
                pairs_by_gender=dict(collections.Counter(rec[files[i]]['gender'] for i, _ in sel_pairs)),
                intents=len({rec[files[i]]['intent'] for i, _ in sel_pairs}),
                pairs_by_intent=dict(collections.Counter(rec[files[i]]['intent'] for i, _ in sel_pairs).most_common()),
                pairs_by_source_split=dict(collections.Counter(f"{rec[files[i]]['split']}+{rec[files[j]]['split']}" for i, j in sel_pairs)))


def main(data, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rec, pairs = qualified_pairs(data)
    files = sorted(rec); idx = {f: i for i, f in enumerate(files)}
    sel = [rec[f] for f in files]
    gd = [(a, b, Da, Db) for a, b, g, dd, Da, Db in pairs if g and dd <= 1.0]
    fd = [(a, b) for a, b, Da, Db in gd if Da and Db]
    rows_nod = [(idx[a], idx[b], True, 0.0) for a, b, _, _ in gd]
    rows_d = [(idx[a], idx[b], True, 0.0) for a, b in fd]
    r_nod = reuse_limited(sel, rows_nod, k=K, lim=1.0)
    r_d = reuse_limited(sel, rows_d, k=K, lim=1.0)
    chosen, opt = ilp_select(len(sel), [(i, j) for i, j, _, _ in rows_d])
    assert opt and len(chosen) == r_d['ilp_max_pairs'], (len(chosen), r_d)
    rep = dict(
        devel_recordings=len(rec), candidate_pairs=len(pairs), same_gender_dur_le_1s=len(gd),
        removed_by_empty_D=len(gd) - len(fd),
        removed_examples=[dict(a=a, b=b, a_values=json.loads(rec[a]['values']), b_values=json.loads(rec[b]['values']))
                          for a, b, Da, Db in gd if not (Da and Db)][:20],
        pass_all_filters=len(fd),
        recordings_in_filtered_edges=len({f for p in fd for f in p}),
        ilp_without_D_filter=r_nod, ilp_with_D_filter=r_d,
        exact_max_pairs=r_d['ilp_max_pairs'], greedy_pairs_lower_bound=r_d['greedy_pairs'],
        diff_vs_v2_197=r_d['ilp_max_pairs'] - 197,
        ilp_solution=describe(chosen, files, rec),
        note='ILP optimum is unique in size, not in selection; distributions describe one optimal selection.')
    (out / 'capacity_summary.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
