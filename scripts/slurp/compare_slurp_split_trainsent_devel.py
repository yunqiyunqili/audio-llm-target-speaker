"""Counts for the chosen split design (2026-11-03, round 6). Counts only: no audio, no manifests.

Speakers: test fraction 0.4, devel 0.2 of the remaining known-gender speakers, same seed (2026) and
  gender-stratified order as scripts/slurp/compare_slurp_speaker_splits.py; UNK speakers -> train.
Sentences (deviation from the official split, decided by the user):
  test  = official test sentences (unchanged)
  devel = ~15% of the official TRAIN sentences, held out from train; official devel sentences unused
  train = remaining official train sentences
  Held-out unit = normalised sentence text (lower/strip), so identical texts under different slurp_ids
  move together. Stratified by (intent, entity-type multiset): within each stratum (sorted), text units
  are sorted, shuffled with random.Random(2026) and round(0.15 * n) are taken. Strata are built from the
  rule-A official-train recordings (all speakers). Text units outside rule A are not needed.
Then each split keeps only its own speakers; cross-set identical texts vs test are removed from
train/devel (evaluation side kept). Reuse limit K=3 (exact ILP) for same-gender pairs with |dd|<=1 s and <=2 s.
Usage: python -I scripts/slurp/compare_slurp_split_trainsent_devel.py ANNOT_DIR DURATIONS_CSV
"""
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_slurp_speaker_heldout import SEED, SPLITS, load  # noqa: E402
from compare_slurp_speaker_splits import RULE_A, assign, cross, describe, norm, reuse_limited  # noqa: E402

F_TEST, F_DEVEL_SENT = 0.4, 0.15


def tvd(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = sum(ca.values()), sum(cb.values())
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def main(annot, durations):
    recs = load(annot, durations)
    spk, names = assign(recs, F_TEST)
    trainA = [r for r in recs if r['split'] == 'train' and RULE_A(r)]
    unit_stratum = {}
    for r in trainA:
        unit_stratum.setdefault(norm(r['sentence']), set()).add((r['intent'], r['types']))
    multi = sum(len(v) > 1 for v in unit_stratum.values())
    strata = collections.defaultdict(list)
    for u, st in unit_stratum.items():
        strata[sorted(st)[0]].append(u)
    rng = random.Random(SEED); dev_units = set()
    for key in sorted(strata):
        units = sorted(strata[key]); rng.shuffle(units)
        dev_units |= set(units[:round(F_DEVEL_SENT * len(units))])
    keep = dict(
        test=[r for r in recs if r['split'] == 'test' and r['usrid'] in spk['test'] and RULE_A(r)],
        devel=[r for r in trainA if norm(r['sentence']) in dev_units and r['usrid'] in spk['devel']],
        train=[r for r in trainA if norm(r['sentence']) not in dev_units and r['usrid'] in spk['train']])
    before = cross(keep)
    t_test = {norm(r['sentence']) for r in keep['test']}
    removed = {s: sum(norm(r['sentence']) in t_test for r in keep[s]) for s in ('train', 'devel')}
    for s in ('train', 'devel'):
        keep[s] = [r for r in keep[s] if norm(r['sentence']) not in t_test]
    rep = dict(seed=SEED, test_fraction=F_TEST, devel_sentence_fraction=F_DEVEL_SENT,
               speakers_assigned={s: len(v) for s, v in spk.items()}, heldout_names=names,
               text_units_train_ruleA=len(unit_stratum), text_units_in_multiple_strata=multi,
               strata=len(strata), devel_text_units=len(dev_units),
               cross_before=before, removed_vs_test=removed, cross_after=cross(keep), splits={})
    for s in SPLITS:
        d, rows = describe(keep[s])
        for lim in (1, 2):
            d[f'reuse_k3_same_gender_dur_le_{lim}s'] = reuse_limited(keep[s], rows, lim=lim)
        rep['splits'][s] = d
        print(s, json.dumps(d), flush=True)
    sent = {s: {r['slurp_id']: r for r in keep[s]}.values() for s in SPLITS}
    rep['distribution_tvd_sentences'] = {
        f'train_vs_{s}': dict(intent=tvd([r['intent'] for r in sent['train']], [r['intent'] for r in sent[s]]),
                              entity_types=tvd([r['types'] for r in sent['train']], [r['types'] for r in sent[s]]))
        for s in ('devel', 'test')}
    rep['intents'] = {s: len({r['intent'] for r in keep[s]}) for s in SPLITS}
    print(json.dumps({k: rep[k] for k in ('speakers_assigned', 'devel_text_units', 'strata', 'text_units_in_multiple_strata',
                                          'cross_before', 'removed_vs_test', 'cross_after', 'distribution_tvd_sentences',
                                          'intents')}), flush=True)
    (PROJ / 'results/slurp/split_trainsent_devel_f0.4.json').write_text(json.dumps(rep, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
