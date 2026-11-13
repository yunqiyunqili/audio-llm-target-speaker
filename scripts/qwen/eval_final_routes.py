"""Final input-path adapter. Imports frozen C5/C6 scoring and consistency checks.
--lora runs P2-only c6-real on the fixed adapter outputs; never changes rules.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_c6_real as real
import eval_c6_delta_spatialnet as delta
import eval_snapshot_downstream as downstream
from eval_c4_v1 import score, summ
import split_cfg  # 2026-11-12 SLURP test switch; devel default unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selected', type=Path, required=True)
    ap.add_argument('--other', type=Path, required=True)
    ap.add_argument('--selection-dir', type=Path, required=True)
    ap.add_argument('--c5', type=Path, required=True)
    ap.add_argument('--delta', type=Path, required=True)
    ap.add_argument('--real', type=Path, required=True)
    ap.add_argument('--lora', action='store_true')
    ap.add_argument('--natural', choices=('cv_dev93', 'test_eval92'), default='cv_dev93')  # test_eval92: descriptive only (freeze §0-5)
    a = ap.parse_args()
    if a.natural != 'cv_dev93':
        assert split_cfg.SPLIT == 'test'; print('NATURAL', split_cfg.set_natural(a.natural), flush=True)
    assert not a.c5.exists() and not a.real.exists()
    test = split_cfg.SPLIT == 'test'
    note = ('FINAL SpatialNet ep100; SLURP TEST (docs/slurp_test_freeze_v1.md); ' if test else 'FINAL SpatialNet ep100; devel only; ') + ('fixed audio LoRA step440' if a.lora else 'P2 and Whisper+T2 baselines')
    def load(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        N = sum(1 for _ in open(real.MANIFEST)) if test else 200
        assert len(rows) == N and len({r['file'] for r in rows}) == N
        return {r['file']: r for r in rows}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(real.MANIFEST))}
    L = {g['group']: g for g in json.load(open(split_cfg.LIST))['groups']}
    selection = {r['key']: r for r in map(json.loads, open(a.selection_dir / 'selection.jsonl'))}
    assert set(selection) == set(man)
    for route, invert in [(a.selected, False), (a.other, True)]:
        for channel in (['P2'] if a.lora else ['P2', 'W', 'T2']):
            raw = load(route / channel / 'raw.jsonl'); assert set(raw) == set(man)
            if channel == 'P2':
                assert all(raw[k]['stream'] == (1 - selection[k]['stream'] if invert else selection[k]['stream']) for k in man)
    if a.lora:
        stats = summ(score(load(a.selected / 'P2/raw.jsonl'), L, man))
        a.c5.parent.mkdir(parents=True, exist_ok=True)
        a.c5.write_text(json.dumps({'note': note, 'models': {'spatialnet': {'P2': stats}}}, indent=2) + '\n')
    elif test:  # C6-delta is not run on the SLURP test (freeze §3); C5 scoring unchanged
        downstream.main(str(a.c5), f'spatialnet={a.selected}')
        obj = json.loads(a.c5.read_text()); obj['note'] = note
        a.c5.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    else:
        assert not a.delta.exists()
        downstream.main(str(a.c5), f'spatialnet={a.selected}')
        delta.main(a.selected, a.other, a.selection_dir / 'selection.jsonl', a.selection_dir / 'per_sample.jsonl',
                   real.MANIFEST, real.O / 'c6_delta_v1/eval_c6_delta.json', a.delta)
        for file in [a.c5, a.delta / 'eval_c6_delta_spatialnet.json']:
            obj = json.loads(file.read_text()); obj['note'] = note
            file.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    c5 = json.loads(a.c5.read_text())['models']['spatialnet']
    routes = {'P2': (a.selected / 'P2/raw.jsonl', a.other / 'P2/raw.jsonl')}
    stored = {'P2': lambda: c5['P2']}
    if not a.lora:
        routes['WT2'] = (a.selected / 'T2/raw.jsonl', a.other / 'T2/raw.jsonl')
        stored['WT2'] = lambda: c5['WhisperT2']
    real.FE = {'spatialnet': dict(sel=a.selection_dir / 'selection.jsonl',
                                ref=('snap', a.selection_dir / 'per_sample.jsonl'), routes=routes)}
    real.STORED = {'spatialnet': stored}
    real.main(a.real, final_note=note, spatial_delta_reference=None if test else a.delta / 'eval_c6_delta_spatialnet.json')


if __name__ == '__main__':
    main()
