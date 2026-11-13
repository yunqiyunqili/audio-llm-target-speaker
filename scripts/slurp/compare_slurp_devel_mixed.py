"""Round 7 counts (only statistics; no manifests, no audio).

Speakers (unchanged): test 40%, devel 20% of the remaining known-gender speakers, seed 2026, gender-stratified
  (same order as scripts/slurp/compare_slurp_speaker_splits.py); UNK -> train.
Sentences:
  test  = official test sentences (unchanged)
  devel = ALL official devel sentences + a fraction f of official TRAIN sentences, f in {0.25,0.30,0.35,0.40}
  train = the remaining official train sentences
  Held-out unit = normalised sentence text (lower/strip); identical texts move together.
  Stratified by intent only: within each intent (sorted), text units are sorted, shuffled with
  random.Random(2026) (one RNG over all intents in sorted order) and round(f * n) are taken.
  Units come from rule-A official-train recordings (all speakers).
Cross-set identical texts (evaluation side kept): text in test -> removed from train/devel;
  text in devel -> removed from train.
Reuse limit K=3: exact ILP (scipy milp/HiGHS) for same-gender pairs with |dd| <= 1 s and <= 2 s.
Usage: python -I scripts/slurp/compare_slurp_devel_mixed.py ANNOT_DIR DURATIONS_CSV
"""
import collections
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_slurp_speaker_heldout import SEED, SPLITS, load  # noqa: E402
from compare_slurp_speaker_splits import RULE_A, assign, cross, describe, norm, reuse_limited  # noqa: E402

F_TEST = 0.4
FRACTIONS = (0.25, 0.30, 0.35, 0.40)


def tvd(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = sum(ca.values()), sum(cb.values())
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def main(annot, durations):
    recs = load(annot, durations)
    spk, names = assign(recs, F_TEST)
    A = [r for r in recs if RULE_A(r)]
    trainA = [r for r in A if r['split'] == 'train']
    unit_intents = collections.defaultdict(set)
    for r in trainA:
        unit_intents[norm(r['sentence'])].add(r['intent'])
    strata = collections.defaultdict(list)
    for u, ints in unit_intents.items():
        strata[sorted(ints)[0]].append(u)
    rep = dict(seed=SEED, test_fraction=F_TEST, speakers_assigned={s: len(v) for s, v in spk.items()},
               heldout_names=names, text_units_train_ruleA=len(unit_intents),
               text_units_with_multiple_intents=sum(len(v) > 1 for v in unit_intents.values()),
               intent_strata=len(strata), fractions={})
    for f in FRACTIONS:
        rng = random.Random(SEED); dev_units = set()
        for key in sorted(strata):
            units = sorted(strata[key]); rng.shuffle(units)
            dev_units |= set(units[:round(f * len(units))])
        keep = dict(
            test=[r for r in A if r['split'] == 'test' and r['usrid'] in spk['test']],
            devel=[r for r in A if r['usrid'] in spk['devel'] and
                   (r['split'] == 'devel' or (r['split'] == 'train' and norm(r['sentence']) in dev_units))],
            train=[r for r in trainA if norm(r['sentence']) not in dev_units and r['usrid'] in spk['train']])
        before = cross(keep)
        t_test = {norm(r['sentence']) for r in keep['test']}
        n_dev0 = len(keep['devel'])
        keep['devel'] = [r for r in keep['devel'] if norm(r['sentence']) not in t_test]
        t_dev = {norm(r['sentence']) for r in keep['devel']}
        n_tr0 = len(keep['train'])
        keep['train'] = [r for r in keep['train'] if norm(r['sentence']) not in t_test | t_dev]
        e = dict(devel_text_units_from_train=len(dev_units), cross_before=before,
                 removed=dict(devel_vs_test=n_dev0 - len(keep['devel']), train_vs_test_or_devel=n_tr0 - len(keep['train'])),
                 cross_after=cross(keep), splits={})
        for s in SPLITS:
            d, rows = describe(keep[s])
            d['intents'] = len({r['intent'] for r in keep[s]})
            d['devel_source'] = dict(collections.Counter(r['split'] for r in keep[s])) if s == 'devel' else None
            for lim in (1, 2):
                d[f'reuse_k3_same_gender_dur_le_{lim}s'] = reuse_limited(keep[s], rows, lim=lim)
            e['splits'][s] = d
        sent = {s: list({r['slurp_id']: r for r in keep[s]}.values()) for s in SPLITS}
        e['tvd_sentences'] = {f'train_vs_{s}': dict(
            intent=tvd([r['intent'] for r in sent['train']], [r['intent'] for r in sent[s]]),
            entity_types=tvd([r['types'] for r in sent['train']], [r['types'] for r in sent[s]])) for s in ('devel', 'test')}
        rep['fractions'][str(f)] = e
        print(f, json.dumps({k: v for k, v in e.items()}), flush=True)
    (PROJ / 'results/slurp/split_devel_mixed_f0.4.json').write_text(json.dumps(rep, indent=2) + '\n')
    print(json.dumps({k: rep[k] for k in ('speakers_assigned', 'text_units_train_ruleA',
                                          'text_units_with_multiple_intents', 'intent_strata')}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
