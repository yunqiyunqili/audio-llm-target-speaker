"""SLURP annotation statistics per split (train/devel/test), headset (close-talk) recordings only.

Rules
  A: recording status == 'correct' and wer == 0 and ent_wer == 0  (fields of recordings[] in *.jsonl)
  B: recording status == 'correct'
Speaker = metadata.json recordings[file].usrid (pattern GenderLanguage-Id, e.g. FE-488).

Pairs (unordered, same split, both recordings pass the rule):
  same intent, same entity-type multiset (non-empty), different entity values
  (sorted (type, value) multiset differs), different speakers (usrid).
Counted exactly with inclusion-exclusion inside each (intent, type-multiset) group:
  pairs = all - same_speaker - same_values + same_speaker_and_same_values
Note: in this release every jsonl recording has status=='correct', and ent_wer is null
(None) for utterances without entities, so rule A excludes them (None != 0).
Also reported: same-gender subset (both F or both M; UNK excluded) and how many recordings
have at least one valid partner.
Usage: python -I scripts/slurp/slurp_annotation_stats.py ANNOT_DIR OUT_JSON
"""
import collections
import json
import sys
from itertools import combinations

SPLITS = ('train', 'devel', 'test')


def gender_of(usrid):
    """usrid = GenderLanguage-Id; Gender in {F, M, UNK}."""
    p = usrid.split('-')[0]
    return 'UNK' if p.startswith('UNK') else p[0]


def c2(n):
    return n * (n - 1) // 2


def entity_values(rec):
    surf = {t['id']: t['surface'] for t in rec['tokens']}
    return tuple(sorted((e['type'], ' '.join(surf[i] for i in e['span']).lower()) for e in rec['entities']))


def pair_stats(items):
    """items: list of (speaker, values, gender). Returns exact counts of valid pairs."""
    n = len(items)
    spk = collections.Counter(s for s, _, _ in items)
    val = collections.Counter(v for _, v, _ in items)
    both = collections.Counter((s, v) for s, v, _ in items)
    total = c2(n) - sum(map(c2, spk.values())) - sum(map(c2, val.values())) + sum(map(c2, both.values()))
    same_gender = 0
    for g in ('F', 'M'):
        sub = [(s, v) for s, v, gg in items if gg == g]
        same_gender += (c2(len(sub)) - sum(map(c2, collections.Counter(s for s, _ in sub).values()))
                        - sum(map(c2, collections.Counter(v for _, v in sub).values()))
                        + sum(map(c2, collections.Counter(sub).values())))
    # recordings with >=1 partner: exists item with different speaker and different values
    with_partner = 0
    for s, v, _ in items:
        partners = n - spk[s] - val[v] + both[(s, v)]
        with_partner += partners > 0
    return total, same_gender, with_partner


def main(annot, out):
    meta = json.load(open(f'{annot}/metadata.json'))
    rec_meta = {}
    for sid, m in meta.items():
        for f, r in m['recordings'].items():
            rec_meta[f] = dict(r, slurp_id=sid)
    usr = collections.defaultdict(set)
    for r in rec_meta.values():
        g, num = r['usrid'].split('-', 1)
        usr[num].add(g)
    report = dict(annotation_dir=annot, rules=dict(A='status==correct & wer==0 & ent_wer==0', B='status==correct'),
                  usrid_gender_prefixes=dict(collections.Counter(r['usrid'].split('-')[0] for r in rec_meta.values())),
                  usrid_numbers_with_conflicting_prefix=sorted(k for k, v in usr.items() if len(v) > 1),
                  splits={})
    speakers_by_split = {}
    for split in SPLITS:
        recs = [json.loads(l) for l in open(f'{annot}/{split}.jsonl')]
        headset, missing_meta, status_mismatch = [], 0, 0
        all_files = 0
        for r in recs:
            vals = entity_values(r)
            types = tuple(sorted(t for t, _ in vals))
            for x in r['recordings']:
                all_files += 1
                if not x['file'].endswith('-headset.flac'):
                    continue
                m = rec_meta.get(x['file'])
                if m is None:
                    missing_meta += 1; continue
                status_mismatch += m['status'] != x['status']
                headset.append(dict(file=x['file'], slurp_id=r['slurp_id'], intent=r['intent'], types=types,
                                    values=vals, status=x['status'], wer=x['wer'], ent_wer=x['ent_wer'],
                                    usrid=m['usrid'], gender=gender_of(m['usrid'])))
        rules = {'A': lambda h: h['status'] == 'correct' and h['wer'] == 0 and h['ent_wer'] == 0,
                 'B': lambda h: h['status'] == 'correct'}
        sr = dict(slurp_records=len(recs), recordings_all=all_files, headset_recordings=len(headset),
                  headset_missing_in_metadata=missing_meta, headset_status_mismatch_jsonl_vs_metadata=status_mismatch,
                  headset_speakers=len({h['usrid'] for h in headset}),
                  headset_ent_wer_null=sum(h['ent_wer'] is None for h in headset),
                  headset_without_entities=sum(not h['types'] for h in headset),
                  headset_status_counts=dict(collections.Counter(h['status'] for h in headset)))
        for name, rule in rules.items():
            sel = [h for h in headset if rule(h)]
            groups = collections.defaultdict(list)
            for h in sel:
                if h['types']:
                    groups[(h['intent'], h['types'])].append((h['usrid'], h['values'], h['gender']))
            pairs = same_g = partner = 0
            for items in groups.values():
                p, g, w = pair_stats(items); pairs += p; same_g += g; partner += w
            spk = {h['usrid'] for h in sel}
            speakers_by_split.setdefault(name, {})[split] = spk
            sr[name] = dict(recordings=len(sel), speakers=len(spk),
                            speakers_by_prefix=dict(collections.Counter(u.split('-')[0] for u in spk)),
                            speakers_by_gender=dict(collections.Counter(gender_of(u) for u in spk)),
                            recordings_ent_wer_null=sum(h['ent_wer'] is None for h in sel),
                            recordings_ent_wer_null_with_entities=sum(h['ent_wer'] is None and bool(h['types']) for h in sel),
                            recordings_without_entities=sum(not h['types'] for h in sel),
                            groups_intent_typeset=len(groups),
                            pairable_pairs=pairs, pairable_pairs_same_gender=same_g,
                            recordings_with_at_least_one_partner=partner)
        report['splits'][split] = sr
    report['speaker_overlap_between_splits'] = {
        name: {f'{a}&{b}': len(d[a] & d[b]) for a, b in combinations(SPLITS, 2)} for name, d in speakers_by_split.items()}
    json.dump(report, open(out, 'w'), indent=2)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
