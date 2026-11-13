"""mix_spec_v1 mixture generation (docs/mix_spec_v1.md) for configs/mix_v1_devel100_v2.json (list v2): 100 groups x 2 swaps.
Per group (SMS-WSJ cv_dev93 RIR example, 8 kHz, 6 mics; SLURP devel close-talk, 16 kHz):
  1. s = resample_poly(slurp_16k, 1, 2) (same scipy call as C1b); assert length == n_8k in the list.
  2. rir_start[pos] = min over channels of the first sample with |h| > 0.1 * max|h| (SMS-WSJ get_rir_start_sample,
     re-implemented); early RIR = h[:rir_start + 400] (50 ms), rest zero (SMS-WSJ scenario_map_fn default).
  3. image = extract_piece(fftconvolve(s, h), offset - rir_start, T) for full and early RIR (SMS-WSJ alignment);
     offset follows the speaker (list), identical in both swaps.
  4. Level: per swap, each source is scaled so that its mic-0 EARLY image has active-frame RMS = LEVEL_DBFS; active
     frames = 256-sample (32 ms) non-overlapping frames whose power is >= (peak frame power - 40 dB). Hence SIR = 0 dB
     on the mic-0 early image in both swaps. The same gain is applied to all channels and to full/early images.
  5. Noise (SMS-WSJ: white Gaussian, 6 ch, SNR ~ U(20,30) dB w.r.t. the power of the clean mixture over all channels
     and samples): ONE noise array per group (np.random.RandomState(list seed)), scaled to the group SNR w.r.t. the
     mean clean-mixture power of the two swaps, added identically to both swaps.
  6. observation = sum of full images + noise (6 ch, 8 kHz); mic0_16k = resample_poly(observation[0], 2, 1)[:2T].
Outputs per (group, swap) in OUT/g{group:03d}_s{swap}/ (float32 wav): obs_6ch_8k.wav, mic0_16k.wav,
  early_A_6ch_8k.wav, early_B_6ch_8k.wav; plus OUT/manifest.jsonl. --dry-run: everything is computed, NO audio is
  written; OUT/dryrun_checks.jsonl + OUT/dryrun_summary.json (achieved SIR/SNR, peaks, sizes) only.
Recorded only (never used to adjust gains): full-image SIR on mic 0 (target vs interferer, each over its own active
  frames, same definition) and its swap-1 minus swap-2 difference per group; overlap = frames active in both sources'
  mic-0 full images / frames active in either; per-sample SNR; peaks and clipped-sample counts (|x| >= 1) of the
  6-ch observation and of mic0_16k.
Usage: python -I scripts/sms_wsj/mix_slurp_v1.py LIST_JSON AUDIO_DIR OUT_DIR [--dry-run]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

EARLY = int(8000 * 0.05)
LEVEL_DBFS = -30.0
FRAME, ACTIVE_DB = 256, 40.0


def rir_start(h, level_ratio=0.1):
    starts = []
    for c in h:
        a = np.abs(c); m = int(np.argmax(a))
        starts.append(int(np.argmax(a[:m + 1] > level_ratio * a[m])))
    return min(starts)


def extract_piece(x, offset, T):
    x = x[..., -offset:] if offset < 0 else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(offset, 0)])
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, T - x.shape[-1])]) if x.shape[-1] < T else x[..., :T]


def frame_activity(x):
    n = len(x) // FRAME
    p = (x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1)
    return p, p >= p.max() * 10 ** (-ACTIVE_DB / 10)


def active_power(x):
    p, act = frame_activity(x)
    return float(p[act].mean()), int(act.sum()), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('list_json', type=Path); ap.add_argument('audio_dir', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    L = json.load(open(a.list_json))
    a.out.mkdir(parents=True, exist_ok=False)
    checks, nbytes, t0 = [], 0, time.perf_counter()
    for g in L['groups']:
        T = g['T_8k']
        src = {}
        for k in 'AB':
            x, sr = sf.read(a.audio_dir / g[k]['file'], dtype='float64')
            assert sr == 16000 and x.ndim == 1 and len(x) == g[k]['n_16k'], (g[k]['file'], sr, x.shape)
            s = resample_poly(x, 1, 2); assert len(s) == g[k]['n_8k'], (len(s), g[k]['n_8k'])
            src[k] = s
        h, rs = [], []
        for pos in (0, 1):
            hp, sr = sf.read(Path(g['rir']['rir_dir']) / f'h_{pos}.wav', dtype='float64', always_2d=True)
            assert sr == 8000 and hp.shape[1] == 6, (sr, hp.shape)
            h.append(hp.T); rs.append(rir_start(hp.T))
        he = [hh.copy() for hh in h]
        for pos in (0, 1):
            he[pos][:, rs[pos] + EARLY:] = 0
        swaps = []
        for sw in g['swaps']:
            pos = dict(A=sw['A_pos'], B=sw['B_pos'])
            img, early, gain, info = {}, {}, {}, {}
            for k in 'AB':
                p = pos[k]; off = g[k]['offset_8k'] - rs[p]
                full = extract_piece(fftconvolve(src[k][None], h[p], axes=-1), off, T)
                ear = extract_piece(fftconvolve(src[k][None], he[p], axes=-1), off, T)
                P, nact, nfr = active_power(ear[0])
                gain[k] = 10 ** (LEVEL_DBFS / 20) / np.sqrt(P)
                img[k], early[k] = full * gain[k], ear * gain[k]
                info[k] = dict(pos=p, rir_start=rs[p], gain=gain[k], active_frames=nact, frames=nfr)
            clean = img['A'] + img['B']
            tgt = sw['target']; itf = 'B' if tgt == 'A' else 'A'
            sir = 10 * np.log10(active_power(early[tgt][0])[0] / active_power(early[itf][0])[0])
            sir_full = 10 * np.log10(active_power(img[tgt][0])[0] / active_power(img[itf][0])[0])
            _, aA = frame_activity(img['A'][0]); _, aB = frame_activity(img['B'][0])
            overlap = float((aA & aB).sum() / max((aA | aB).sum(), 1))
            swaps.append(dict(sw=sw, clean=clean, early=early, info=info, sir=sir, sir_full=sir_full, overlap=overlap))
        rng = np.random.RandomState(g['noise']['seed'])
        noise = rng.normal(size=(6, T))
        pc = np.mean([np.mean(s['clean'] ** 2) for s in swaps])
        noise *= np.sqrt(pc / 10 ** (g['noise']['snr_db'] / 10) / np.mean(noise ** 2))
        for s in swaps:
            obs = s['clean'] + noise
            mic0 = resample_poly(obs[0], 2, 1)[:2 * T]
            snr = 10 * np.log10(np.mean(s['clean'] ** 2) / np.mean(noise ** 2))
            target = s['sw']['target']
            rec = dict(group=g['group'], swap=s['sw']['swap'], target=target, target_file=g[target]['file'],
                       interferer_file=g['B' if target == 'A' else 'A']['file'], theta_question=g['rir']['theta1'],
                       T_8k=T, sir_mic0_early_db=float(s['sir']), sir_mic0_full_db=float(s['sir_full']),
                       sir_full_swap1_minus_swap2_db=float(swaps[0]['sir_full'] - swaps[1]['sir_full']),
                       overlap_active_frames=s['overlap'], snr_db_achieved=float(snr), snr_db_group=g['noise']['snr_db'],
                       peak_obs=float(np.abs(obs).max()), peak_mic0_16k=float(np.abs(mic0).max()),
                       clipped_obs=int((np.abs(obs) >= 1).sum()), clipped_mic0_16k=int((np.abs(mic0) >= 1).sum()),
                       sources=s['info'])
            checks.append(rec)
            files = dict(obs_6ch_8k=(obs.T, 8000), mic0_16k=(mic0, 16000), early_A_6ch_8k=(s['early']['A'].T, 8000),
                         early_B_6ch_8k=(s['early']['B'].T, 8000))
            nbytes += sum(arr.size * 4 + 44 for arr, _ in files.values())
            if not a.dry_run:
                d = a.out / f"g{g['group']:03d}_s{s['sw']['swap']}"; d.mkdir()
                for name, (arr, sr) in files.items():
                    sf.write(d / f'{name}.wav', arr.astype(np.float32), sr, subtype='FLOAT')
                rec['dir'] = str(d)
    name = 'dryrun' if a.dry_run else 'manifest'
    with open(a.out / (f'{name}_checks.jsonl' if a.dry_run else 'manifest.jsonl'), 'w') as fh:
        for r in checks:
            fh.write(json.dumps(r) + '\n')
    sir = np.array([r['sir_mic0_early_db'] for r in checks]); snr = np.array([r['snr_db_achieved'] for r in checks])
    pk = np.array([r['peak_obs'] for r in checks])
    pct = lambda k, f=None: {q: float(np.percentile([(f or (lambda r: r[k]))(r) for r in checks], v))  # noqa: E731
                             for q, v in (('min', 0), ('p5', 5), ('median', 50), ('p95', 95), ('max', 100))}
    summ = dict(dry_run=a.dry_run, samples=len(checks), seconds=time.perf_counter() - t0, audio_bytes=nbytes,
                audio_written=not a.dry_run, level_dbfs=LEVEL_DBFS,
                sir_db=dict(min=float(sir.min()), max=float(sir.max())), snr_db=dict(min=float(snr.min()), max=float(snr.max())),
                peak_obs=dict(max=float(pk.max()), n_ge_1=int((pk >= 1).sum())),
                peak_mic0_16k_max=float(max(r['peak_mic0_16k'] for r in checks)),
                clipped_samples_obs=int(sum(r['clipped_obs'] for r in checks)),
                clipped_samples_mic0_16k=int(sum(r['clipped_mic0_16k'] for r in checks)),
                sir_full_db=pct('sir_mic0_full_db'), sir_full_swap_diff_db=pct('sir_full_swap1_minus_swap2_db'),
                sir_full_swap_diff_abs_db=pct(None, lambda r: abs(r['sir_full_swap1_minus_swap2_db'])),
                overlap_active_frames=pct('overlap_active_frames'), snr_db_per_sample=pct('snr_db_achieved'),
                snr_minus_group_db=pct(None, lambda r: r['snr_db_achieved'] - r['snr_db_group']),
                rir_start=dict(min=min(v['rir_start'] for r in checks for v in r['sources'].values()),
                               max=max(v['rir_start'] for r in checks for v in r['sources'].values())))
    (a.out / f'{name}_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
