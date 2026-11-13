"""SLURP speaker-held-out version on top of the official sentence (slurp_id) split.

Population: close-talk (-headset) recordings of SLURP commit 8eb1654.
Excluded up front: audio-1502309360-headset.flac (one file attached to two different sentences).
Held-out speakers: gender-stratified ~20% of known-gender (F/M) speakers, seed 2026
  (per gender: sort usrids, random.Random(2026).shuffle, take round(0.2 * n)).
  train/devel: drop all recordings of held-out speakers.
  test: keep only recordings of held-out speakers (official test sentences).
  UNK-gender speakers are never held out (so they appear only in train/devel) and never
  enter same-gender pairs.
Rules:
  A (main):        status == correct and wer == 0 and ent_wer == 0   (not relaxed; ent_wer null fails)
  S (sensitivity): status == correct, no wer/ent_wer requirement
Pairing (unchanged): same split, same intent, same non-empty entity-type multiset, different
  (type,value) multiset, different usrid. Utterances without entities never pair (counted).
Outputs (repo):   configs/slurp_heldout_speakers_seed2026.json, results/slurp/heldout_stats.json
Outputs (shared): DATA_OUT/{A,S}_{split}_recordings.csv, DATA_OUT/{A,S}_{split}_pairs.csv.gz
Usage: python -I scripts/slurp/make_slurp_speaker_heldout.py ANNOT_DIR DURATIONS_CSV DATA_OUT
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

SEED, FRACTION = 2026, 0.2
SPLITS = ('train', 'devel', 'test')
EXCLUDE = {'audio-1502309360-headset.flac': 'same file attached to slurp_id 15959 and 16142 (different sentences)'}
PROJ = Path(__file__).resolve().parents[2]


def gender_of(usrid):
    p = usrid.split('-')[0]
    return 'UNK' if p.startswith('UNK') else p[0]


def load(annot, durations):
    meta = json.load(open(f'{annot}/metadata.json'))
    usr = {f: r['usrid'] for v in meta.values() for f, r in v['recordings'].items()}
    dur = {(r['split'], int(r['slurp_id']), r['file']): float(r['seconds']) for r in csv.DictReader(open(durations))}
    recs = []
    for split in SPLITS:
        for line in open(f'{annot}/{split}.jsonl'):
            r = json.loads(line)
            surf = {t['id']: t['surface'] for t in r['tokens']}
            vals = tuple(sorted((e['type'], ' '.join(surf[i] for i in e['span']).lower()) for e in r['entities']))
            for x in r['recordings']:
                f = x['file']
                if not f.endswith('-headset.flac') or f in EXCLUDE:
                    continue
                recs.append(dict(split=split, slurp_id=r['slurp_id'], sentence=r['sentence'], intent=r['intent'],
                                 types='|'.join(t for t, _ in vals), values=json.dumps(vals), file=f, usrid=usr[f],
                                 gender=gender_of(usr[f]), status=x['status'], wer=x['wer'], ent_wer=x['ent_wer'],
                                 seconds=dur[(split, r['slurp_id'], f)]))
    return recs


def choose_heldout(recs):
    spk = sorted({(r['gender'], r['usrid']) for r in recs})
    chosen, pools = {}, {}
    for g in ('F', 'M'):
        pool = sorted(u for gg, u in spk if gg == g)
        rng = random.Random(SEED); order = pool[:]; rng.shuffle(order)
        k = round(FRACTION * len(pool))
        chosen[g] = sorted(order[:k]); pools[g] = len(pool)
    return chosen, pools, sum(1 for g, _ in spk if g == 'UNK')


def pairs_in_split(sel):
    """Exact enumeration inside (intent, types) groups. Returns counts + pair rows."""
    groups = collections.defaultdict(list)
    for i, r in enumerate(sel):
        if r['types']:
            groups[(r['intent'], r['types'])].append(i)
    out = collections.Counter(); rows = []; partner = set()
    for idx in groups.values():
        if len(idx) < 2:
            continue
        spk = np.array([sel[i]['usrid'] for i in idx]); val = np.array([sel[i]['values'] for i in idx])
        gen = np.array([sel[i]['gender'] for i in idx]); sec = np.array([sel[i]['seconds'] for i in idx])
        a, b = np.triu_indices(len(idx), 1)
        ok = (spk[a] != spk[b]) & (val[a] != val[b])
        a, b = a[ok], b[ok]
        same_g = (gen[a] == gen[b]) & (gen[a] != 'UNK')
        dd = np.abs(sec[a] - sec[b])
        out['pairs'] += len(a); out['pairs_same_gender'] += int(same_g.sum())
        out['pairs_dur_le_1s'] += int((dd <= 1).sum()); out['pairs_dur_le_2s'] += int((dd <= 2).sum())
        out['pairs_same_gender_dur_le_1s'] += int((same_g & (dd <= 1)).sum())
        out['pairs_same_gender_dur_le_2s'] += int((same_g & (dd <= 2)).sum())
        ia = np.array(idx)[a]; ib = np.array(idx)[b]
        partner.update(ia.tolist()); partner.update(ib.tolist())
        rows.extend(zip(ia.tolist(), ib.tolist(), same_g.tolist(), dd.round(3).tolist()))
    out['recordings_with_partner'] = len(partner)
    out['groups'] = len(groups)
    return dict(out), rows


def main(annot, durations, data_out):
    data_out = Path(data_out); data_out.mkdir(parents=True, exist_ok=True)
    recs = load(annot, durations)
    chosen, pools, n_unk = choose_heldout(recs)
    held = set(chosen['F']) | set(chosen['M'])
    (PROJ/'configs').mkdir(exist_ok=True)
    (PROJ/'configs/slurp_heldout_speakers_seed2026.json').write_text(json.dumps(dict(
        seed=SEED, fraction=FRACTION, annotations=annot, population='headset recordings, known-gender usrid',
        known_gender_speakers=pools, unk_speakers_never_held_out=n_unk,
        heldout=chosen, heldout_total=len(held), excluded_files=EXCLUDE), indent=2) + '\n')
    split_recs = {s: [r for r in recs if r['split'] == s and ((r['usrid'] in held) == (s == 'test'))] for s in SPLITS}
    rules = {'A': lambda r: r['status'] == 'correct' and r['wer'] == 0 and r['ent_wer'] == 0,
             'S': lambda r: r['status'] == 'correct'}
    report = dict(seed=SEED, heldout_total=len(held), heldout_by_gender={g: len(v) for g, v in chosen.items()},
                  known_gender_pool=pools, excluded_files=EXCLUDE,
                  recordings_removed_from_train_devel=sum(1 for r in recs if r['split'] != 'test' and r['usrid'] in held),
                  recordings_removed_from_test=sum(1 for r in recs if r['split'] == 'test' and r['usrid'] not in held),
                  rules={'A': 'status==correct & wer==0 & ent_wer==0', 'S': 'status==correct'}, splits={}, checks={})
    keep = {}
    for name, rule in rules.items():
        for s in SPLITS:
            sel = [r for r in split_recs[s] if rule(r)]
            keep[(name, s)] = sel
            counts, rows = pairs_in_split(sel)
            spk = {r['usrid'] for r in sel}
            report['splits'].setdefault(s, {})[name] = dict(
                recordings=len(sel), speakers=len(spk),
                speakers_by_gender=dict(collections.Counter(gender_of(u) for u in spk)),
                sentences=len({r['slurp_id'] for r in sel}),
                recordings_without_entities_not_pairable=sum(not r['types'] for r in sel),
                seconds_total_hours=sum(r['seconds'] for r in sel) / 3600, **counts)
            with open(data_out/f'{name}_{s}_recordings.csv', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(sel[0])); w.writeheader(); w.writerows(sel)
            with gzip.open(data_out/f'{name}_{s}_pairs.csv.gz', 'wt', newline='') as fh:
                w = csv.writer(fh); w.writerow(['file_a', 'file_b', 'same_gender', 'abs_duration_diff_s'])
                for i, j, g, d in rows:
                    w.writerow([sel[i]['file'], sel[j]['file'], int(g), d])
    for name in rules:
        sid = {s: {r['slurp_id'] for r in keep[(name, s)]} for s in SPLITS}
        txt = {s: {r['sentence'].lower().strip() for r in keep[(name, s)]} for s in SPLITS}
        spk = {s: {r['usrid'] for r in keep[(name, s)]} for s in SPLITS}
        report['checks'][name] = {f'{a}&{b}': dict(slurp_id=len(sid[a] & sid[b]), sentence_text=len(txt[a] & txt[b]),
                                                   speakers=len(spk[a] & spk[b]))
                                  for a, b in combinations(SPLITS, 2)}
    (PROJ/'results/slurp/heldout_stats.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
