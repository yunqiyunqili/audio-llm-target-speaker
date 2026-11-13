"""SLURP split v2 (2026-11-04 decision): stats, checks and -- only if thresholds are met -- formal lists.

Intent label = f"{scenario}_{action}" everywhere (the jsonl `intent` field disagrees for some sentences).
Speakers: test 40%, devel 20% of the remaining known-gender speakers, gender-stratified, seed 2026
  (scripts/slurp/compare_slurp_speaker_splits.py: assign); UNK -> train.
Sentences: test = official test; devel = ALL official devel + 35% of official train text units
  (stratified by scenario_action only; identical normalised texts move together; random.Random(2026) over
  sorted strata); train = remaining official train. Cross-set identical texts vs test removed from train/devel,
  vs devel removed from train. Rule A. Pairing rule unchanged (pairs_in_split), now grouped by scenario_action.
Thresholds for writing formal lists: test >= 500 and devel >= 100 pairs with reuse<=3, same gender, |dd|<=1 s.
Formal lists (if written): DATA_OUT/{split}_recordings.csv, DATA_OUT/{split}_candidate_pairs.csv.gz (ALL candidate
  pairs, not a final draw), configs/slurp_split_v2_seed2026.json (seed, speakers, devel text units).
No audio is generated and no final experiment pairs are drawn.
Usage: python -I scripts/slurp/make_slurp_split_v2.py ANNOT_DIR DURATIONS_CSV DATA_OUT
"""
import collections
import csv
import gzip
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_slurp_speaker_heldout import EXCLUDE, SEED, SPLITS, load, pairs_in_split  # noqa: E402
from compare_slurp_speaker_splits import RULE_A, assign, cross, describe, norm, reuse_limited  # noqa: E402

F_TEST, F_DEVEL_TRAIN = 0.4, 0.35
MIN_TEST, MIN_DEVEL = 500, 100


def tvd(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = sum(ca.values()), sum(cb.values())
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def label_audit(annot):
    """Intent field vs scenario_action, and entity-type label inventory per split."""
    out = dict(intent_field_mismatch={}, entity_types={}, scenario_action_labels={})
    et = {s: collections.Counter() for s in SPLITS}; mism = collections.Counter(); sa_lab = {}
    for s in SPLITS:
        n = 0
        for line in open(f'{annot}/{s}.jsonl'):
            r = json.loads(line); sa = f"{r['scenario']}_{r['action']}"
            n += r['intent'] != sa; mism[(r['intent'], sa)] += r['intent'] != sa
            sa_lab.setdefault(s, set()).add(sa)
            for e in r['entities']:
                et[s][e['type']] += 1
        out['intent_field_mismatch'][s] = n
    out['intent_field_mismatch_pairs'] = {f'{a} -> {b}': c for (a, b), c in mism.most_common() if c}
    out['scenario_action_labels'] = {s: len(v) for s, v in sa_lab.items()}
    out['scenario_action_not_in_train'] = {s: sorted(sa_lab[s] - sa_lab['train']) for s in ('devel', 'test')}
    allt = set().union(*[set(c) for c in et.values()])
    out['entity_types'] = dict(
        n_per_split={s: len(c) for s, c in et.items()}, n_total=len(allt),
        not_in_train={s: sorted(set(et[s]) - set(et['train'])) for s in ('devel', 'test')},
        not_lowercase_snake=sorted(t for t in allt if t != t.strip().lower() or ' ' in t or '-' in t),
        rare_in_train_lt5=sorted((t, c) for t, c in et['train'].items() if c < 5),
        counts_train=dict(et['train'].most_common()))
    return out


def main(annot, durations, data_out):
    audit = label_audit(annot)
    recs = load(annot, durations)
    sa = {}
    for s in SPLITS:
        for line in open(f'{annot}/{s}.jsonl'):
            j = json.loads(line); sa[j['slurp_id']] = f"{j['scenario']}_{j['action']}"
    for r in recs:
        r['intent_field'] = r['intent']; r['intent'] = sa[r['slurp_id']]
    spk, names = assign(recs, F_TEST)
    A = [r for r in recs if RULE_A(r)]
    trainA = [r for r in A if r['split'] == 'train']
    unit_int = collections.defaultdict(set)
    for r in trainA:
        unit_int[norm(r['sentence'])].add(r['intent'])
    strata = collections.defaultdict(list)
    for u, ints in unit_int.items():
        strata[sorted(ints)[0]].append(u)
    rng = random.Random(SEED); dev_units = set()
    for key in sorted(strata):
        units = sorted(strata[key]); rng.shuffle(units)
        dev_units |= set(units[:round(F_DEVEL_TRAIN * len(units))])
    keep = dict(test=[r for r in A if r['split'] == 'test' and r['usrid'] in spk['test']],
                devel=[r for r in A if r['usrid'] in spk['devel'] and
                       (r['split'] == 'devel' or (r['split'] == 'train' and norm(r['sentence']) in dev_units))],
                train=[r for r in trainA if norm(r['sentence']) not in dev_units and r['usrid'] in spk['train']])
    before = cross(keep)
    t_test = {norm(r['sentence']) for r in keep['test']}
    n0 = {s: len(v) for s, v in keep.items()}
    keep['devel'] = [r for r in keep['devel'] if norm(r['sentence']) not in t_test]
    t_dev = {norm(r['sentence']) for r in keep['devel']}
    keep['train'] = [r for r in keep['train'] if norm(r['sentence']) not in t_test | t_dev]
    rep = dict(seed=SEED, test_fraction=F_TEST, devel_train_sentence_fraction=F_DEVEL_TRAIN, intent_label='scenario_action',
               label_audit=audit, speakers_assigned={s: len(v) for s, v in spk.items()}, heldout_names=names,
               text_units_train_ruleA=len(unit_int), text_units_multi_intent=sum(len(v) > 1 for v in unit_int.values()),
               intent_strata=len(strata), devel_text_units_from_train=len(dev_units), excluded_files=EXCLUDE,
               cross_before=before, removed={s: n0[s] - len(keep[s]) for s in SPLITS}, cross_after=cross(keep), splits={})
    rows_by = {}
    for s in SPLITS:
        d, rows = describe(keep[s]); rows_by[s] = rows
        d['intents'] = len({r['intent'] for r in keep[s]})
        if s == 'devel':
            d['source'] = dict(collections.Counter(r['split'] for r in keep[s]))
        for lim in (1, 2):
            d[f'reuse_k3_same_gender_dur_le_{lim}s'] = reuse_limited(keep[s], rows, lim=lim)
        rep['splits'][s] = d
        print(s, json.dumps(d), flush=True)
    sent = {s: list({r['slurp_id']: r for r in keep[s]}.values()) for s in SPLITS}
    rep['tvd_sentences'] = {f'train_vs_{s}': dict(intent=tvd([r['intent'] for r in sent['train']], [r['intent'] for r in sent[s]]),
                                                  entity_types=tvd([r['types'] for r in sent['train']], [r['types'] for r in sent[s]]))
                            for s in ('devel', 'test')}
    t_ok = rep['splits']['test']['reuse_k3_same_gender_dur_le_1s']
    d_ok = rep['splits']['devel']['reuse_k3_same_gender_dur_le_1s']
    met = (t_ok['ilp_optimal'] and d_ok['ilp_optimal'] and t_ok['ilp_max_pairs'] >= MIN_TEST and d_ok['ilp_max_pairs'] >= MIN_DEVEL)
    rep['thresholds'] = dict(min_test=MIN_TEST, min_devel=MIN_DEVEL, test=t_ok['ilp_max_pairs'], devel=d_ok['ilp_max_pairs'], met=met)
    if met:
        out = Path(data_out); out.mkdir(parents=True, exist_ok=False)
        files = {}
        for s in SPLITS:
            sel = keep[s]
            p = out / f'{s}_recordings.csv'
            with open(p, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(sel[0])); w.writeheader(); w.writerows(sel)
            q = out / f'{s}_candidate_pairs.csv.gz'
            with gzip.open(q, 'wt', newline='') as fh:
                w = csv.writer(fh); w.writerow(['file_a', 'file_b', 'same_gender', 'abs_duration_diff_s'])
                for i, j, g, dd in rows_by[s]:
                    w.writerow([sel[i]['file'], sel[j]['file'], int(g), dd])
            files[s] = dict(recordings=str(p), candidate_pairs=str(q), n_recordings=len(sel), n_candidate_pairs=len(rows_by[s]))
        (PROJ / 'configs/slurp_split_v2_seed2026.json').write_text(json.dumps(dict(
            seed=SEED, test_fraction=F_TEST, devel_speaker_fraction_of_remaining=0.2, devel_train_sentence_fraction=F_DEVEL_TRAIN,
            intent_label='scenario_action', rule='A', annotations=annot, speakers={s: sorted(v) for s, v in spk.items()},
            speakers_with_recordings={s: sorted({r['usrid'] for r in keep[s]}) for s in SPLITS},
            devel_text_units_from_train=sorted(dev_units), excluded_files=EXCLUDE, files=files,
            note='candidate pairs = all pairs satisfying the pairing rule; no final draw; no audio'), indent=1) + '\n')
        rep['formal_lists'] = files
    (PROJ / 'results/slurp/split_v2_stats.json').write_text(json.dumps(rep, indent=2) + '\n')
    print('THRESHOLDS', json.dumps(rep['thresholds']), 'cross_after', json.dumps(rep['cross_after']), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
