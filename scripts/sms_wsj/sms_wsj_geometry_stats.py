"""Spatial-geometry statistics for all SMS-WSJ splits (read-only on sms_wsj.json).

Per mixture: room-frame azimuth of each speaker (atan2(dy,dx) from array centre),
azimuth relative to centre->mic0, inter-speaker azimuth separation (0..180 deg),
elevation, distance, T60. Outputs JSON + per-mixture CSV into --out.

Run on a compute node, e.g.
  srun -A <SLURM_ACCOUNT> -p <CPU_PARTITION> -c 2 --mem=8G -t 00:20:00 python -I scripts/sms_wsj/sms_wsj_geometry_stats.py --out results/sms_wsj_geometry
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DB = '<SEPARATION_DATA_ROOT>/data/sms_wsj/sms_wsj.json'
SPLITS = ('train_si284', 'cv_dev93', 'test_eval92')
SEP_BUCKETS = [(0, 15), (15, 30), (30, 60), (60, 180.0001)]  # degrees, [lo, hi)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def chi2_uniform(counts):
    counts = np.asarray(counts, float)
    exp = counts.sum() / len(counts)
    stat = float(((counts - exp) ** 2 / exp).sum())
    dof = len(counts) - 1
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, dof))
    except ImportError:
        p = None
    return dict(chi2=stat, dof=dof, p_value=p)


def rayleigh(angles_deg):
    """Rayleigh test of circular uniformity; R = mean resultant length."""
    a = np.radians(angles_deg)
    n = len(a)
    C, S = np.cos(a).sum(), np.sin(a).sum()
    R = math.hypot(C, S) / n
    z = n * R * R
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - (n * R) ** 2)) - (1 + 2 * n))  # Zar (1999) approx.
    return dict(mean_resultant_length=R, mean_direction_deg=math.degrees(math.atan2(S, C)), z=z,
                p_value=min(1.0, p))


def describe(x):
    x = np.asarray(x, float)
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()),
                min=float(q[0]), p5=float(q[1]), p25=float(q[2]), median=float(q[3]),
                p75=float(q[4]), p95=float(q[5]), max=float(q[6]))


def hist(x, width, lo, hi):
    edges = np.arange(lo, hi + 1e-9, width)
    c, _ = np.histogram(x, bins=edges)
    return {f'[{edges[i]:g},{edges[i+1]:g})': int(c[i]) for i in range(len(c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = json.load(open(DB))['datasets']
    report = dict(source=DB, azimuth_convention='atan2(dy,dx) of source minus array centre, room x/y axes; '
                  'relative = minus azimuth of centre->mic0; separation = |wrap(az1-az0)| in [0,180]',
                  separation_buckets_deg=[list(b) for b in SEP_BUCKETS], splits={})
    for split in SPLITS:
        rows = []
        for key in sorted(ds[split]):
            e = ds[split][key]
            src = np.asarray(e['source_position'], float)
            mic = np.asarray(e['sensor_position'], float)
            assert src.shape == (3, 2) and mic.shape == (3, 6), key
            c = mic.mean(1)
            ref = math.degrees(math.atan2(*(mic[:, 0] - c)[[1, 0]]))
            r = dict(id=key, t60=float(e['sound_decay_time']), rir_dir=str(Path(e['audio_path']['rir'][0]).parent))
            az = []
            for s in range(2):
                d = src[:, s] - c
                a = math.degrees(math.atan2(d[1], d[0])); az.append(a)
                r[f'az_room_{s}'] = a
                r[f'az_rel_mic0_{s}'] = wrap(a - ref)
                r[f'elev_{s}'] = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
                r[f'dist_{s}'] = float(np.linalg.norm(d))
            r['separation'] = abs(wrap(az[1] - az[0]))
            r['mic0_dir_room'] = ref
            rows.append(r)
        with open(args.out / f'{split}_per_mixture.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        az_room = np.array([r[f'az_room_{s}'] for r in rows for s in range(2)])
        az_rel = np.array([r[f'az_rel_mic0_{s}'] for r in rows for s in range(2)])
        sep = np.array([r['separation'] for r in rows])
        t60 = np.array([r['t60'] for r in rows])
        sector = hist(az_room, 30, -180, 180)
        buckets = {f'[{lo},{min(hi,180):g})' if hi < 180 else f'[{lo},180]':
                   int(((sep >= lo) & (sep < hi)).sum()) for lo, hi in SEP_BUCKETS}
        report['splits'][split] = dict(
            mixtures=len(rows), speaker_instances=len(az_room),
            unique_rir_dirs=len({r['rir_dir'] for r in rows}),
            separation_buckets=buckets,
            separation=describe(sep), separation_hist_15deg=hist(sep, 15, 0, 180),
            t60_s=describe(t60), t60_hist_50ms=hist(t60, 0.05, 0.0, 0.7),
            az_room=dict(sector_counts_30deg=sector, chi2_vs_uniform=chi2_uniform(list(sector.values())),
                         rayleigh=rayleigh(az_room),
                         max_over_min_sector=max(sector.values()) / max(1, min(sector.values()))),
            az_rel_mic0=dict(sector_counts_30deg=hist(az_rel, 30, -180, 180),
                             chi2_vs_uniform=chi2_uniform(list(hist(az_rel, 30, -180, 180).values())),
                             rayleigh=rayleigh(az_rel)),
            mic0_dir_room=dict(rayleigh=rayleigh([r['mic0_dir_room'] for r in rows])),
            elevation_deg=describe([r[f'elev_{s}'] for r in rows for s in range(2)]),
            distance_m=describe([r[f'dist_{s}'] for r in rows for s in range(2)]),
        )
        print(split, json.dumps(report['splits'][split]['separation_buckets']), flush=True)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('WROTE', args.out / 'summary.json')


if __name__ == '__main__':
    main()
