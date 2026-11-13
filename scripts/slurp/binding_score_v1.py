"""Key-entity-value binding score v1 (docs/binding_score_v1.md) on the CLEAN C1-extension outputs (job 22010997).
Clean-speech baseline only: the target's own clean output is checked against (target, interferer) value sets.
No mixture audio exists; no pair flip is computed.
Pair (T, I): devel candidate pair (split_v2_seed2026: same split, same intent, same entity-type multiset, values differ,
  different speakers) with same gender and |duration diff| <= 1 s, and D_T, D_I both non-empty, where
  D_T = T - I, D_I = I - T as MULTISETS of normalize_v2(entity value) (types ignored; both sides have the same type multiset).
Main (parsed entity values only, full-value match, multiset):
  target_all = every value of D_T is among the hyp values (with multiplicity); interferer_any = any value of D_I is a hyp value.
  categories: only_target (target_all & not interferer_any), only_interferer, both, neither.
  Also: target_any (>= 1 value of D_T hit), target hit rate = |D_T ∩ H| / |D_T|.
Diagnostic (P0, P1 only): same, with "hit" = value occurs as a whole-word substring of the normalised transcript.
Aggregation: per target recording over all its qualified interferers (proportions, rates; min/max of the hit rate over
  interferers), then the unweighted mean over recordings. Targets = the tier-1 recordings of configs/c1_ext_devel100.json.
Usage: python -I scripts/slurp/binding_score_v1.py DATA_DIR RUN_DIR OUT_DIR
"""
import collections
import csv
import gzip
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen'))
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import parse  # noqa: E402

VERSION = 'binding-score-v1'


def vals(entities):
    return collections.Counter(normalize_v2(v) for _, v in entities)


def qualified_pairs(data, split='devel'):
    """All devel candidate pairs with flags; returns recordings dict and list of (a, b, same_g, dd, D_a, D_b).
    split='test' (2026-11-12, docs/slurp_test_freeze_v1.md §4) reads the test files instead; devel default unchanged."""
    with open(Path(data) / f'{split}_recordings.csv', newline='') as fh:
        rec = {r['file']: r for r in csv.DictReader(fh)}
    V = {f: vals(json.loads(r['values'])) for f, r in rec.items()}
    out = []
    with gzip.open(Path(data) / f'{split}_candidate_pairs.csv.gz', 'rt', newline='') as fh:
        for p in csv.DictReader(fh):
            a, b = p['file_a'], p['file_b']
            out.append((a, b, p['same_gender'] == '1', float(p['abs_duration_diff_s']), V[a] - V[b], V[b] - V[a]))
    return rec, out


def judge(D_T, D_I, H):
    hitc = sum((D_T & H).values()); n = sum(D_T.values())
    t_all = hitc == n; t_any = hitc > 0; i_any = any(H[v] > 0 for v in D_I)
    cat = 'both' if t_all and i_any else 'only_target' if t_all else 'only_interferer' if i_any else 'neither'
    return dict(cat=cat, t_all=t_all, t_any=t_any, rate=hitc / n)


def transcript_counter(text, values):
    t = f' {normalize_v2(text)} '
    return collections.Counter({v: t.count(f' {v} ') for v in values})


def main(data, run, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rec, pairs = qualified_pairs(data)
    partners = collections.defaultdict(list)
    for a, b, g, dd, Da, Db in pairs:
        if g and dd <= 1.0 and Da and Db:
            partners[a].append((b, Da, Db)); partners[b].append((a, Db, Da))
    cfg = json.load(open(PROJ / 'configs/c1_ext_devel100.json'))
    targets = [it['file'] for it in cfg['items'] if it['tier'] == 1]
    rep = dict(version=VERSION, run=str(run), note='clean-speech baseline; target output only; no mixtures; no pair flip',
               targets_tier1=len(targets), targets_without_qualified_interferer=sum(not partners[f] for f in targets),
               pairs_total=sum(len(partners[f]) for f in targets),
               interferers_per_target=dict(min=min(len(partners[f]) for f in targets if partners[f]),
                                           max=max(len(partners[f]) for f in targets)),
               conditions={})
    per_rec_rows = []
    for cond in ('P0', 'P1', 'T0'):
        raw = {json.loads(l)['file']: json.loads(l) for l in open(Path(run) / cond / 'raw.jsonl')}
        modes = ['entities'] + (['transcript_diagnostic'] if cond != 'T0' else [])
        rep['conditions'][cond] = {}
        for mode in modes:
            recs = []
            for f in targets:
                if not partners[f]:
                    continue
                obj, st = parse(raw[f]['raw_output'])
                js = []
                for b, D_T, D_I in partners[f]:
                    if mode == 'entities':
                        H = vals(obj['entities']) if obj else collections.Counter()
                    else:
                        H = transcript_counter(obj['transcript'] if obj else '', set(D_T) | set(D_I))
                    js.append(judge(D_T, D_I, H))
                c = collections.Counter(j['cat'] for j in js); n = len(js)
                r = dict(file=f, condition=cond, mode=mode, n_interferers=n, parse_status=st,
                         **{k: c[k] / n for k in ('only_target', 'only_interferer', 'both', 'neither')},
                         target_all=sum(j['t_all'] for j in js) / n, target_any=sum(j['t_any'] for j in js) / n,
                         hit_rate_mean=sum(j['rate'] for j in js) / n, hit_rate_min=min(j['rate'] for j in js),
                         hit_rate_max=max(j['rate'] for j in js))
                recs.append(r); per_rec_rows.append(r)
            m = len(recs)
            mean = lambda k: sum(r[k] for r in recs) / m  # noqa: E731
            rep['conditions'][cond][mode] = dict(
                recordings=m, pairs=sum(r['n_interferers'] for r in recs),
                **{f'mean_{k}': mean(k) for k in ('only_target', 'only_interferer', 'both', 'neither', 'target_all', 'target_any',
                                                   'hit_rate_mean', 'hit_rate_min', 'hit_rate_max')},
                recordings_with_hit_rate_range_gt0=sum(r['hit_rate_max'] > r['hit_rate_min'] for r in recs),
                hit_rate_range_over_recordings=dict(min_of_min=min(r['hit_rate_min'] for r in recs),
                                                    max_of_max=max(r['hit_rate_max'] for r in recs)))
    (out / 'binding_summary.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(out / 'binding_per_recording.jsonl', 'w') as fh:
        for r in per_rec_rows:
            fh.write(json.dumps(r) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
