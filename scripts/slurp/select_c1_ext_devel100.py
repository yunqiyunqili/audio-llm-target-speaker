"""Select 100 DEVEL recordings (split_v2_seed2026) for the C1 extension (P0 / P1 / T0) + 3 fixed TRAIN examples. List only.

Population: DATA/devel_recordings.csv (v2 devel; rule A; intent = scenario_action). Test is never read.
Priority tiers (a recording's best tier counts):
  1 = in >= 1 devel candidate pair with same gender and |duration diff| <= 1 s (the formal-pair criterion)
  2 = in >= 1 devel candidate pair (any gender / duration)
  3 = in no candidate pair
Stratification by scenario_action: quota_s = proportional to the number of DISTINCT slurp_ids of s in devel,
  largest-remainder rounding to N=100. Within a stratum: recordings sorted by file, random.Random(2026) shuffle,
  then stable-sorted by tier; greedy pick with distinct slurp_id (one recording per sentence).
  If a stratum cannot fill its quota, the shortfall is filled from the remaining recordings of all strata in the
  same (tier, shuffled) order.
Examples for P1: 3 recordings' TEXT + labels from DATA/train_recordings.csv (v2 train): sorted by file, same seed
  shuffle, greedy: >= 1 entity, distinct scenarios, no shared entity type, entity values without " 's" tokens.
Usage: python -I scripts/slurp/select_c1_ext_devel100.py DATA_DIR
"""
import collections
import csv
import gzip
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
SEED, N = 2026, 100


def read(p):
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))


def main(data):
    data = Path(data)
    dev = read(data / 'devel_recordings.csv')
    tier = {r['file']: 3 for r in dev}
    with gzip.open(data / 'devel_candidate_pairs.csv.gz', 'rt', newline='') as fh:
        for p in csv.DictReader(fh):
            t = 1 if p['same_gender'] == '1' and float(p['abs_duration_diff_s']) <= 1.0 else 2
            for f in (p['file_a'], p['file_b']):
                tier[f] = min(tier[f], t)
    sent = collections.defaultdict(set)
    for r in dev:
        sent[r['intent']].add(r['slurp_id'])
    tot = sum(len(v) for v in sent.values())
    raw = {s: N * len(v) / tot for s, v in sent.items()}
    quota = {s: int(x) for s, x in raw.items()}
    for s in sorted(raw, key=lambda s: (-(raw[s] - quota[s]), s))[:N - sum(quota.values())]:
        quota[s] += 1
    rng = random.Random(SEED)
    order = sorted(dev, key=lambda r: r['file']); rng.shuffle(order)
    order.sort(key=lambda r: tier[r['file']])           # stable: shuffled order kept within each tier
    pick, used = [], set()
    for s in sorted(quota):
        k = 0
        for r in order:
            if k == quota[s]:
                break
            if r['intent'] == s and r['slurp_id'] not in used:
                pick.append(r); used.add(r['slurp_id']); k += 1
    short = N - len(pick)
    for r in order:
        if len(pick) == N:
            break
        if r['slurp_id'] not in used:
            pick.append(r); used.add(r['slurp_id'])
    items = [dict(file=r['file'], slurp_id=int(r['slurp_id']), usrid=r['usrid'], gender=r['gender'], seconds=float(r['seconds']),
                  sentence=r['sentence'], intent=r['intent'], entities=json.loads(r['values']), source_split=r['split'],
                  tier=tier[r['file']]) for r in pick]
    # fixed examples from v2 train
    tr = sorted(read(data / 'train_recordings.csv'), key=lambda r: r['file']); random.Random(SEED).shuffle(tr)
    ex, scen, types = [], set(), set()
    for r in tr:
        ents = json.loads(r['values']); ty = {t for t, _ in ents}
        if not ents or r['intent'].split('_')[0] in scen or ty & types or any(" '" in v for _, v in ents):
            continue
        ex.append(dict(file=r['file'], slurp_id=int(r['slurp_id']), sentence=r['sentence'], intent=r['intent'], entities=ents))
        scen.add(r['intent'].split('_')[0]); types |= ty
        if len(ex) == 3:
            break
    c20 = {it['file'] for it in json.load(open(PROJ / 'configs/c1_semantic_devel20.json'))['items']}
    stats = dict(n=len(items), strata_in_devel=len(sent), strata_in_sample=len({i['intent'] for i in items}),
                 quota_shortfall_filled_globally=short, tier_counts=dict(collections.Counter(i['tier'] for i in items)),
                 devel_tier_counts=dict(collections.Counter(tier.values())), speakers=len({i['usrid'] for i in items}),
                 gender=dict(collections.Counter(i['gender'] for i in items)),
                 source_split=dict(collections.Counter(i['source_split'] for i in items)),
                 with_entities=sum(bool(i['entities']) for i in items), n_entities=sum(len(i['entities']) for i in items),
                 overlap_with_c1_devel20=len(c20 & {i['file'] for i in items}),
                 seconds_min=min(i['seconds'] for i in items), seconds_max=max(i['seconds'] for i in items))
    (PROJ / 'configs/c1_ext_devel100.json').write_text(json.dumps(dict(
        seed=SEED, n=N, source=str(data / 'devel_recordings.csv'), rule=__doc__.split('Usage')[0], stats=stats,
        quota=quota, examples_train=ex, items=items), indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(stats))
    for e in ex:
        print('EXAMPLE', e['intent'], '|', e['sentence'], '|', e['entities'])


if __name__ == '__main__':
    main(sys.argv[1])
