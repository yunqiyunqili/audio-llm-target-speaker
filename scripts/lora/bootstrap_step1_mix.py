"""Pre-specified paired stratified-group bootstrap for C4-direct/C3 differences.
Plan v2 §4: B=2000, seed=2026, resample groups within angle strata, keep both
swaps together. Descriptive intervals; frozen point-estimate gates unchanged.
Usage: python -I -B scripts/lora/bootstrap_step1_mix.py NEW_OUTPUT_DIR
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_step1_checks as e


def main(destination):
    groups = json.load(open(e.LIST))['groups']
    gids = sorted(g['group'] for g in groups)
    random.Random(2026).shuffle(gids)
    selection = set(gids[:50])
    strata = {g['group']: g['rir']['stratum'] for g in groups}
    report = {'B': 2000, 'seed': 2026, 'weighting': 'equal-sample; paired groups stratified by angle', 'conditions': {}}
    for cond, job in [('c4direct', '22027859'), ('c3', '22027860')]:
        lp = e.one(f'lora_audio_mix/lr1e-4/{job}/step440/{cond}/raw.jsonl')
        a, b = e.load([lp]), e.load([e.BASE[cond]])
        assert len(a) == 200 and set(a) == set(b)
        ar, br = e.mix_rows(a, groups), e.mix_rows(b, groups)
        result = {}
        for half, keep in [('all', lambda g: True), ('selection', lambda g: g in selection), ('report', lambda g: g not in selection)]:
            files = sorted(f for f in ar if keep(ar[f]['group']))
            group_files = {g: [f for f in files if ar[f]['group'] == g] for g in sorted({ar[f]['group'] for f in files})}
            assert all(len(fs) == 2 for fs in group_files.values())
            ordered = sorted(group_files)
            by_stratum = [[i for i, g in enumerate(ordered) if strata[g] == s] for s in sorted(set(strata.values()))]
            rng = np.random.default_rng(2026)
            draws = np.array([np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_stratum if ix]) for _ in range(2000)])
            output = {'n': len(files), 'n_groups': len(ordered)}
            for metric, value in [('target_any', lambda r: int(r['t_any'])), ('only_interferer', lambda r: int(r['cat'] == 'only_interferer'))]:
                delta = np.array([sum(value(ar[f]) - value(br[f]) for f in group_files[g]) / 2 for g in ordered])
                output[metric] = {'delta': float(delta.mean()), 'ci95': np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()}
            result[half] = output
        report['conditions'][cond] = result
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    (out / 'bootstrap.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
