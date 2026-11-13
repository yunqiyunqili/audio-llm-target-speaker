"""Merge sharded SLURP-test outputs of ONE route into one directory (docs/slurp_test_freeze_v1.md; CPU, no model).

Every part directory holds config.json and <COND>/raw.jsonl for the same conditions. raw.jsonl lines are concatenated in
part order (parts are contiguous slices of the same sorted item list, so the result equals an unsharded run's order);
config.json fields must be identical across parts except per-job provenance (part, slurm_job_id, node, gpu, n_part) and
audio_sha256 (merged; keys must be disjoint). Nothing is rescored or modified. Writes OUT/<COND>/raw.jsonl,
OUT/config.json (+ parts list) and OUT/merge.json (sha256 of every input and output file).
Usage: python -I scripts/lora/merge_test_parts.py OUT PART_DIR [PART_DIR ...]
"""
import hashlib
import json
import sys
from pathlib import Path

PER_JOB = {'part', 'slurm_job_id', 'node', 'gpu', 'n_part', 'git_dirty_files'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out, *parts):
    out, parts = Path(out), [Path(p) for p in parts]
    assert parts and not out.exists()
    conds = sorted(d.name for d in parts[0].iterdir() if d.is_dir())
    assert conds and all(sorted(d.name for d in p.iterdir() if d.is_dir()) == conds for p in parts), conds
    cfg = [json.loads((p / 'config.json').read_text()) for p in parts]
    assert all(c.get('part', 0) == i and c.get('nparts', 1) == len(parts) for i, c in enumerate(cfg)), 'parts must be 0..N-1 in order'
    keys = set(cfg[0]) - PER_JOB - {'audio_sha256'}
    for c in cfg[1:]:
        assert set(c) - PER_JOB - {'audio_sha256'} == keys
        for k in keys:
            assert c[k] == cfg[0][k], k
    merged = {k: cfg[0][k] for k in cfg[0] if k not in PER_JOB and k != 'audio_sha256'}
    if 'audio_sha256' in cfg[0]:
        A = {}
        for c in cfg:
            assert not set(A) & set(c['audio_sha256']); A.update(c['audio_sha256'])
        merged['audio_sha256'] = A
    merged['parts'] = [dict(dir=str(p), **{k: c.get(k) for k in sorted(PER_JOB) if k in c}) for p, c in zip(parts, cfg)]
    out.mkdir(parents=True)
    rec = dict(inputs={}, outputs={}, conditions={})
    for cond in conds:
        (out / cond).mkdir(); lines, seen = [], set()
        for p in parts:
            f = p / cond / 'raw.jsonl'; rec['inputs'][str(f)] = sha(f)
            for line in f.read_text().splitlines():
                r = json.loads(line); assert r['file'] not in seen, r['file']; seen.add(r['file']); lines.append(line)
        (out / cond / 'raw.jsonl').write_text(''.join(x + '\n' for x in lines))
        rec['conditions'][cond] = len(lines); rec['outputs'][str(out / cond / 'raw.jsonl')] = sha(out / cond / 'raw.jsonl')
        if 'audio_sha256' in merged and cond == 'P2':
            assert seen == set(merged['audio_sha256']), cond
    (out / 'config.json').write_text(json.dumps(merged, indent=1, ensure_ascii=False) + '\n')
    rec['outputs'][str(out / 'config.json')] = sha(out / 'config.json')
    (out / 'merge.json').write_text(json.dumps(rec, indent=1) + '\n')
    print('MERGED', json.dumps(rec['conditions']), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
