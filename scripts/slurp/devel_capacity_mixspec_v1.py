"""Devel pair capacity under the mix_spec_v1 pair filter (docs/mix_spec_v1.md). Statistics only; no audio, no draw.
Base: same as scripts/slurp/devel_binding_capacity.py (same gender, |dd| <= 1 s, D_T and D_I non-empty, reuse <= 3).
Extra exclusion: a pair is excluded if for ANY v in D_T and u in D_I (normalize_v2 values):
  (a) stem(v) == stem(u), where stem = per-token: drop trailing "'s"; then "ies"->"y" (len > 4); then drop "es" after
      s/x/z/ch/sh (len > 4); then drop a final "s" not preceded by "s" (len > 3). Simple plural/possessive rule, not Porter.
  (b) the token sequence of v contains the token sequence of u as a contiguous sub-sequence, or vice versa.
Exact maximum: same 0/1 ILP (compare_slurp_speaker_splits.reuse_limited + the selection copy in devel_binding_capacity).
Usage: python -I scripts/slurp/devel_capacity_mixspec_v1.py DATA_DIR OUT_DIR
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binding_score_v1 import qualified_pairs  # noqa: E402
from compare_slurp_speaker_splits import reuse_limited  # noqa: E402
from devel_binding_capacity import K, describe, ilp_select  # noqa: E402


def stem_tok(t):
    t = re.sub(r"'s$", '', t)
    if len(t) > 4 and t.endswith('ies'):
        return t[:-3] + 'y'
    if len(t) > 4 and re.search(r'(s|x|z|ch|sh)es$', t):
        return t[:-2]
    if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
        return t[:-1]
    return t


def stem(v):
    return tuple(stem_tok(t) for t in v.split())


def contains(a, b):
    a, b = a.split(), b.split()
    return any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))


def reason(DT, DI):
    for v in DT:
        for u in DI:
            if stem(v) == stem(u):
                return 'stem', v, u
            if contains(v, u) or contains(u, v):
                return 'contains', v, u
    return None


def main(data, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rec, pairs = qualified_pairs(data)
    files = sorted(rec); idx = {f: i for i, f in enumerate(files)}; sel = [rec[f] for f in files]
    base = [(a, b, Da, Db) for a, b, g, dd, Da, Db in pairs if g and dd <= 1.0 and Da and Db]
    excl, keep = [], []
    for a, b, Da, Db in base:
        r = reason(Da, Db)
        (excl if r else keep).append((a, b, r))
    rows = [(idx[a], idx[b], True, 0.0) for a, b, _ in keep]
    r = reuse_limited(sel, rows, k=K, lim=1.0)
    chosen, opt = ilp_select(len(sel), [(i, j) for i, j, _, _ in rows])
    assert opt and len(chosen) == r['ilp_max_pairs']
    rep = dict(
        base_pairs=len(base), excluded=len(excl), excluded_by_reason=dict(stem=sum(x[2][0] == 'stem' for x in excl),
                                                                          contains=sum(x[2][0] == 'contains' for x in excl)),
        excluded_pairs=[dict(a=rec[a]['sentence'], b=rec[b]['sentence'], reason=x[0], v=x[1], u=x[2]) for a, b, x in excl],
        kept_pairs=len(keep), recordings_in_kept_pairs=len({f for a, b, _ in keep for f in (a, b)}),
        ilp=r, exact_max_pairs=r['ilp_max_pairs'], greedy_pairs_lower_bound=r['greedy_pairs'],
        diff_vs_197=r['ilp_max_pairs'] - 197, ilp_solution=describe(chosen, files, rec),
        note='one optimal selection; size is exact, selection not unique')
    (out / 'capacity_mixspec_v1.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:3])
