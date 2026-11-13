"""Select 20 clean (C1) close-talk recordings from the DEVEL candidates for the Qwen semantic pilot. List only.

Devel candidates (stable for every train-hold-out fraction being compared): rule-A recordings of OFFICIAL devel
sentences spoken by devel speakers (test 40% / devel 20% assignment, seed 2026, gender-stratified), with texts
that also occur in the test candidates removed. Test is never used.
Selection: random.Random(2026) shuffle of the candidates (sorted by file first), then greedy passes with
(max per intent, max per speaker) = (1,1), (1,2), (2,2), (unlimited) until 20 items; slurp_id always distinct.
Intent label = f"{scenario}_{action}" (60 labels in official train). The jsonl `intent` field disagrees with
scenario_action for 229 sentences (e.g. 'query', 'music', 'set'), so it is NOT used here.
Also writes the label inventory (scenario_action intents, entity types) of the OFFICIAL TRAIN split, used in the prompt.
Usage: python -I scripts/slurp/select_c1_semantic_devel.py ANNOT_DIR DURATIONS_CSV
"""
import collections
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_slurp_speaker_heldout import SEED, load  # noqa: E402
from compare_slurp_speaker_splits import RULE_A, assign, norm  # noqa: E402

N = 20


def main(annot, durations):
    recs = load(annot, durations)
    sa = {}
    for sp in ('train', 'devel', 'test'):
        for line in open(f'{annot}/{sp}.jsonl'):
            j = json.loads(line); sa[j['slurp_id']] = f"{j['scenario']}_{j['action']}"
    for r in recs:
        r['intent_field'] = r['intent']; r['intent'] = sa[r['slurp_id']]
    spk, _ = assign(recs, 0.4)
    test_txt = {norm(r['sentence']) for r in recs if r['split'] == 'test' and r['usrid'] in spk['test'] and RULE_A(r)}
    cand = sorted((r for r in recs if r['split'] == 'devel' and r['usrid'] in spk['devel'] and RULE_A(r)
                   and norm(r['sentence']) not in test_txt), key=lambda r: r['file'])
    rng = random.Random(SEED); order = cand[:]; rng.shuffle(order)
    pick, n_int, n_spk = [], collections.Counter(), collections.Counter()
    # passes: (max uses per intent, max uses per speaker); later passes relax the limits
    for max_int, max_spk in ((1, 1), (1, 2), (2, 2), (N, N)):
        for r in order:
            if len(pick) == N:
                break
            if any(p['slurp_id'] == r['slurp_id'] for p in pick) or n_int[r['intent']] >= max_int \
                    or n_spk[r['usrid']] >= max_spk:
                continue
            pick.append(r); n_int[r['intent']] += 1; n_spk[r['usrid']] += 1
    rows = [dict(file=r['file'], slurp_id=r['slurp_id'], usrid=r['usrid'], gender=r['gender'], seconds=r['seconds'],
                 sentence=r['sentence'], intent=r['intent'], intent_field=r['intent_field'], entities=json.loads(r['values'])) for r in pick]
    out = PROJ / 'configs/c1_semantic_devel20.json'
    out.write_text(json.dumps(dict(seed=SEED, n=N, candidates=len(cand),
                                   candidate_speakers=len({c['usrid'] for c in cand}),
                                   rule='A; official devel sentences x devel speakers (f_test=0.4); test texts removed',
                                   items=rows), indent=2, ensure_ascii=False) + '\n')
    intents, etypes = collections.Counter(), collections.Counter()
    for line in open(f'{annot}/train.jsonl'):
        r = json.loads(line); intents[f"{r['scenario']}_{r['action']}"] += 1
        for e in r['entities']:
            etypes[e['type']] += 1
    (PROJ / 'configs/slurp_label_inventory_train.json').write_text(json.dumps(dict(
        source=f'{annot}/train.jsonl (official train, all sentences)', intents=sorted(intents),
        entity_types=sorted(etypes)), indent=2) + '\n')
    print('candidates', len(cand), 'speakers', len({c['usrid'] for c in cand}), 'picked', len(rows),
          'distinct intents', len({r['intent'] for r in rows}), 'speakers', len({r['usrid'] for r in rows}),
          'inventory', len(intents), 'intents', len(etypes), 'entity types')
    for r in rows:
        print(r['intent'], '|', r['sentence'], '|', r['entities'], '|', r['usrid'], round(r['seconds'], 2))


if __name__ == '__main__':
    main(*sys.argv[1:3])
