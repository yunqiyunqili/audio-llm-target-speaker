"""How many more devel groups are possible beyond mix_v1 (configs/mix_v1_devel100_v2.json)? Statistics only; no audio.
Candidates: the 248 mix_spec_v1-filtered devel pairs (same gender, |dd| <= 1 s, D_T/D_I non-empty, no stem/containment)
minus the 100 pairs used by mix_v1. Exact maxima by 0/1 ILP (scipy milp / HiGHS), three readings:
  A  'new set'     : unused pairs, fresh limits (recording <= 3, speaker <= 30 groups), counted on the new set alone
  B  'disjoint'    : unused pairs whose two recordings are both unused by mix_v1; same fresh limits
  C  'additive'    : unused pairs, limits shared with mix_v1 (recording <= 3 - used, speaker <= 30 - used)
For each: max groups; and, subject to that max, the max number of male groups (lexicographic second solve).
Usage: python -I scripts/slurp/mix_v1_remaining_capacity.py DATA_DIR OUT_JSON
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from binding_score_v1 import qualified_pairs  # noqa: E402
from devel_capacity_mixspec_v1 import reason  # noqa: E402


def solve(pairs, rec, cap_rec, cap_spk, fix_total=None, male_obj=False):
    m = len(pairs)
    if m == 0:
        return []
    files = sorted({f for p in pairs for f in p}); fi = {f: i for i, f in enumerate(files)}
    spk = sorted({rec[f]['usrid'] for f in files}); si = {u: i for i, u in enumerate(spk)}
    A = lil_matrix((len(files) + len(spk) + 1, m))
    for j, (a, b) in enumerate(pairs):
        A[fi[a], j] += 1; A[fi[b], j] += 1
        A[len(files) + si[rec[a]['usrid']], j] = 1; A[len(files) + si[rec[b]['usrid']], j] = 1
        A[len(files) + len(spk), j] = 1
    lo = [0] * (len(files) + len(spk)) + [fix_total or 0]
    hi = [cap_rec(f) for f in files] + [cap_spk(u) for u in spk] + [fix_total if fix_total is not None else np.inf]
    c = -np.array([1.0 if rec[a]['gender'] == 'M' else 0.0 for a, _ in pairs]) if male_obj else -np.ones(m)
    res = milp(c=c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(m), bounds=Bounds(0, 1),
               options=dict(time_limit=600))
    assert res.status == 0, res.message
    return [pairs[j] for j in np.flatnonzero(np.round(res.x) == 1)]


def describe(sel, rec):
    spk = collections.Counter(rec[f]['usrid'] for p in sel for f in p)
    return dict(groups=len(sel), male_groups=sum(rec[a]['gender'] == 'M' for a, _ in sel),
                recordings=len({f for p in sel for f in p}), speakers=len(spk), speaker_group_counts=dict(spk.most_common()),
                intents=len({rec[a]['intent'] for a, _ in sel}))


def main(data, out):
    rec, pairs = qualified_pairs(data)
    kept = sorted(tuple(sorted((a, b))) for a, b, g, dd, Da, Db in pairs if g and dd <= 1.0 and Da and Db and not reason(Da, Db))
    L = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']
    used_pairs = {tuple(sorted((g['A']['file'], g['B']['file']))) for g in L}
    use_rec = collections.Counter(f for g in L for f in (g['A']['file'], g['B']['file']))
    use_spk = collections.Counter(g[k]['usrid'] for g in L for k in 'AB')
    unused = [p for p in kept if p not in used_pairs]
    disjoint = [p for p in unused if use_rec[p[0]] == 0 and use_rec[p[1]] == 0]
    cases = dict(A_new_set=(unused, lambda f: 3, lambda u: 30), B_disjoint=(disjoint, lambda f: 3, lambda u: 30),
                 C_additive=(unused, lambda f: 3 - use_rec[f], lambda u: 30 - use_spk[u]))
    rep = dict(kept_pairs=len(kept), used_by_mix_v1=len(used_pairs & set(kept)), unused_pairs=len(unused),
               unused_pairs_disjoint_recordings=len(disjoint), cases={})
    for name, (cand, cr, cs) in cases.items():
        sel = solve(cand, rec, cr, cs)
        best = solve(cand, rec, cr, cs, fix_total=len(sel), male_obj=True) if sel else []
        rep['cases'][name] = dict(candidates=len(cand), max_groups=len(sel), at_max_groups_max_male=describe(best, rec) if best else None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
