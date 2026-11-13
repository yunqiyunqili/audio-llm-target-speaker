"""Audit final separated audio and invoke unchanged v3 selection/scoring functions."""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from eval_snapshot_mix import main as select
import split_cfg  # noqa: E402  (on sys.path via eval_snapshot_mix; 2026-11-12 SLURP test switch)
from make_other_selection import main as other


def main(separation, output):
    sep, out = Path(separation), Path(output)
    expected = {f"g{r['group']:03d}_s{r['swap']}" for r in map(json.loads, open(split_cfg.MANIFEST))}
    rows = [json.loads(line) for line in (sep / 'manifest.jsonl').read_text().splitlines()]
    N = 200 if split_cfg.SPLIT == 'devel' else len(expected)
    assert len(rows) == N and {r['key'] for r in rows} == expected
    assert {p.name for p in sep.iterdir() if p.is_dir()} == expected
    for row in rows:
        for st in (0, 1):
            y, sr = sf.read(sep / row['key'] / f'out{st}.wav')
            assert sr == 8000 and y.shape == (row['samples'],) and np.isfinite(y).all()
    select(sep.parent, str(split_cfg.DIRECT_REFS), out, f'spatialnet={sep.name}', final=True)
    selected = [json.loads(s) for s in (out / 'spatialnet/selection.jsonl').read_text().splitlines()]
    assert len(selected) == N and {r['key'] for r in selected} == expected
    other(out / 'spatialnet/selection.jsonl', out / 'spatialnet/other_selection.jsonl')
    print(f'FINAL_SELECTION_PASS {N} samples / {2 * N} audio files', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
