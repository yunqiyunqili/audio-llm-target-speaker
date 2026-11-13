"""Validate the generated mix_v1 audio (read-only). No front-end, no Qwen.
Checks: sample count (groups x 2 swaps), manifest lines, 4 files per sample, sample rate / channels / length
(obs and early: 8 kHz, 6 ch, T; mic0: 16 kHz, 1 ch, 2T), NaN/Inf, clipped samples (|x| >= 1), agreement of manifest gains
and SNR with the dry-run checks, and a recomputation from the files of the mic-0 early-image level (-30 dBFS target)
and early SIR (0 dB), within float32 tolerance. Draws 3 groups with random.Random(2026) for listening.
Usage: python -I scripts/sms_wsj/check_mix_v1.py LIST_JSON MIX_DIR DRYRUN_CHECKS_JSONL OUT_JSON
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_slurp_v1 import LEVEL_DBFS, active_power  # noqa: E402

FILES = dict(obs_6ch_8k=(8000, 6, 1), mic0_16k=(16000, 1, 2), early_A_6ch_8k=(8000, 6, 1), early_B_6ch_8k=(8000, 6, 1))


def main(list_json, mix_dir, dry, out):
    L = json.load(open(list_json)); mix = Path(mix_dir)
    man = [json.loads(l) for l in open(mix / 'manifest.jsonl')]
    dr = {(r['group'], r['swap']): r for r in map(json.loads, open(dry))}
    T = {g['group']: g['T_8k'] for g in L['groups']}
    dirs = sorted(p for p in mix.iterdir() if p.is_dir())
    problems, lvl, sir, maxdiff_gain, maxdiff_snr = [], [], [], 0.0, 0.0
    for r in man:
        key = (r['group'], r['swap']); d = mix / f"g{r['group']:03d}_s{r['swap']}"
        if Path(r['dir']) != d:
            problems.append(f'{key} dir mismatch')
        x = {}
        for name, (sr, ch, mult) in FILES.items():
            info = sf.info(d / f'{name}.wav')
            if (info.samplerate, info.channels, info.frames, info.subtype) != (sr, ch, mult * T[r['group']], 'FLOAT'):
                problems.append(f'{key} {name} format {info.samplerate} {info.channels} {info.frames} {info.subtype}')
            a, _ = sf.read(d / f'{name}.wav', dtype='float64', always_2d=True)
            if not np.isfinite(a).all():
                problems.append(f'{key} {name} NaN/Inf')
            if (np.abs(a) >= 1).any():
                problems.append(f'{key} {name} clipped {(np.abs(a) >= 1).sum()}')
            x[name] = a
        q = dr[key]
        for k in 'AB':
            maxdiff_gain = max(maxdiff_gain, abs(r['sources'][k]['gain'] - q['sources'][k]['gain']) / q['sources'][k]['gain'])
        maxdiff_snr = max(maxdiff_snr, abs(r['snr_db_achieved'] - q['snr_db_achieved']))
        t = r['target']; i = 'B' if t == 'A' else 'A'
        pt = active_power(x[f'early_{t}_6ch_8k'][:, 0])[0]; pi = active_power(x[f'early_{i}_6ch_8k'][:, 0])[0]
        lvl += [10 * np.log10(pt), 10 * np.log10(pi)]; sir.append(10 * np.log10(pt / pi))
    lvl, sir = np.array(lvl), np.array(sir)
    pick = sorted(random.Random(2026).sample(range(len(L['groups'])), 3))
    listen = [dict(group=gi, theta1=L['groups'][gi]['rir']['theta1'], stratum=L['groups'][gi]['rir']['stratum'],
                   A=L['groups'][gi]['A']['sentence'], B=L['groups'][gi]['B']['sentence'],
                   files={f's{s}': str(mix / f'g{gi:03d}_s{s}' / 'mic0_16k.wav') for s in (1, 2)},
                   targets={'s1': 'A', 's2': 'B'}) for gi in pick]
    rep = dict(expected_samples=2 * len(L['groups']), sample_dirs=len(dirs), manifest_lines=len(man),
               files_per_dir=sorted({len(list(d.glob('*.wav'))) for d in dirs}), problems=problems,
               max_rel_gain_diff_vs_dryrun=maxdiff_gain, max_abs_snr_diff_vs_dryrun_db=maxdiff_snr,
               early_level_dbfs=dict(min=float(lvl.min()), max=float(lvl.max()), target=LEVEL_DBFS),
               early_sir_db=dict(min=float(sir.min()), max=float(sir.max())),
               bytes_on_disk=sum(f.stat().st_size for f in mix.rglob('*') if f.is_file()), listening_sample=listen)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:5])
