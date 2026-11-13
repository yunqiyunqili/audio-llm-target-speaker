"""Post-result descriptive C3 diagnostics; never used for checkpoint selection or gates.
Keep binding-score-v1 unchanged. Additionally separate any-target/any-interferer
hits because its original four categories use target-ALL versus interferer-ANY.
Usage: python -I -B scripts/lora/diagnose_c3_binding.py NEW_OUTPUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_step1_checks as e

CATS = ('only_target', 'only_interferer', 'both', 'neither')
ANY_CATS = ('target_only_any', 'interferer_only_any', 'both_any', 'no_discriminative_hit')


def any_cat(r):
    t, i = r['dt_hits'] > 0, r['di_hits'] > 0
    return 'both_any' if t and i else 'target_only_any' if t else 'interferer_only_any' if i else 'no_discriminative_hit'


def main(destination):
    paths = {'p2': str(e.BASE['c3']), 'lora': e.one('lora_audio_mix/lr1e-4/22027860/step440/c3/raw.jsonl')}
    raw = {name: e.load([path]) for name, path in paths.items()}
    assert len(raw['p2']) == 200 and set(raw['p2']) == set(raw['lora'])
    assert all(raw['p2'][f]['prompt_text'] == raw['lora'][f]['prompt_text'] for f in raw['p2'])
    groups = json.load(open(e.LIST))['groups']
    rows = {name: e.mix_rows(values, groups) for name, values in raw.items()}
    report = {'note': 'Post-result descriptive diagnosis only; no causal attribution; frozen main scores unchanged.',
              'definitions': {'main': 'target ALL / interferer ANY, binding-score-v1',
                              'supplement': 'target ANY / interferer ANY; discriminative normalized values only'},
              'inputs': {name: {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name, path in paths.items()},
              'models': {}}
    for name, rr in rows.items():
        model = {'main': e.msum(list(rr.values())), 'main_counts': {c: sum(r['cat'] == c for r in rr.values()) for c in CATS},
                 'any_hit_counts': {c: sum(any_cat(r) == c for r in rr.values()) for c in ANY_CATS},
                 'entities': e.ents(list(rr.values())), 'no_hit_details': {}}
        nohit = [r for r in rr.values() if any_cat(r) == 'no_discriminative_hit']
        model['no_hit_details'] = {'n': len(nohit), 'zero_predicted_entities': sum(r['n_pred'] == 0 for r in nohit),
                                   'nonempty_but_no_discriminative_hit': sum(r['n_pred'] > 0 for r in nohit),
                                   'parse_failures': sum(r['parse_fail'] for r in nohit)}
        report['models'][name] = model
    report['main_transition_p2_rows_lora_columns'] = {
        a: {b: sum(rows['p2'][f]['cat'] == a and rows['lora'][f]['cat'] == b for f in rows['p2']) for b in CATS} for a in CATS}
    report['any_transition_p2_rows_lora_columns'] = {
        a: {b: sum(any_cat(rows['p2'][f]) == a and any_cat(rows['lora'][f]) == b for f in rows['p2']) for b in ANY_CATS} for a in ANY_CATS}
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    (out / 'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (out / 'paired_samples.jsonl').open('w') as handle:
        for f in sorted(rows['p2']):
            handle.write(json.dumps({'file': f, **{name: rr[f] | {'any_category': any_cat(rr[f])} for name, rr in rows.items()}}) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
