"""Evaluate zero-shot front-ends on mix_v1 exactly as written in docs/frontend_mix_v1_plan.md (frozen before running).
References (mic 0): early image = written early_{A,B}_6ch_8k.wav; full image = recomputed in memory with the mix_slurp_v1
functions and the manifest gains (early is recomputed the same way and must match the files, max |diff| < 1e-6).
SI-SDR = tfgridnet/objectives.py sisdr (imported, read-only). PIT over [target, interferer] as in scores().
Reference matching (C4, offline diagnostic) and direction matching (outputs + 6-mic observation + array geometry only;
PHAT-weighted cross-spectra to mic 0 output, plane-wave azimuth grid search). C5 rule: accuracy >= 0.85 on samples with
separation outside [0,15). Usage (tfgridnet env, CPU):
  python -B scripts/sms_wsj/eval_frontend_mix_v1.py MIX_DIR SEP_ROOT SLURP_AUDIO_DIR OUT_DIR
"""
import sys
sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import fftconvolve, resample_poly, stft  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
MC = Path('<SEPARATION_REPO>/multichannel')
sys.path.insert(0, str(MC)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfgridnet.objectives import sisdr as _sisdr  # noqa: E402
from mix_slurp_v1 import EARLY, extract_piece, rir_start  # noqa: E402

DB = '<SEPARATION_DATA_ROOT>/data/sms_wsj/sms_wsj.json'
FRONTENDS = ('blstm_mvdr', 'fasnet_tac')
C, FMIN, FMAX, NFFT, HOP = 343.0, 80.0, 3900.0, 512, 128
GRID = np.arange(360.0)
THRESH = 0.85
BINS = dict(t60=[(0.2, 0.3), (0.3, 0.4), (0.4, 0.5000001)], overlap=[(0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0000001)],
            sir_full=[(-np.inf, -0.5), (-0.5, 0.5000001), (0.5000001, np.inf)])


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def sisdr(ref, est):
    return float(_sisdr(torch.from_numpy(np.asarray(ref, np.float64)), torch.from_numpy(np.asarray(est, np.float64))))


def doa(y, obs, mic_xy):
    """y: (T,) single-channel estimate at mic 0; obs: (6, T); mic_xy: (6, 2). Returns room azimuth (deg)."""
    f, _, Y = stft(y, fs=8000, window='hann', nperseg=NFFT, noverlap=NFFT - HOP)
    _, _, X = stft(obs, fs=8000, window='hann', nperseg=NFFT, noverlap=NFFT - HOP)
    band = (f >= FMIN) & (f <= FMAX); f = f[band]
    G = (np.conj(Y[band])[None] * X[1:, band]).sum(-1)                      # (5, F)
    G = G / np.maximum(np.abs(G), 1e-12)
    u = np.stack([np.cos(np.radians(GRID)), np.sin(np.radians(GRID))], 1)    # (360, 2)
    tau = -(u @ (mic_xy[1:] - mic_xy[0]).T) / C                              # (360, 5)
    S = np.real(np.einsum('mf,gmf->g', G, np.exp(1j * 2 * np.pi * f[None, None, :] * tau[:, :, None])))
    return float(GRID[int(np.argmax(S))])


def pit(refs, outs):
    v = [np.mean([sisdr(refs[0], outs[p[0]]), sisdr(refs[1], outs[p[1]])]) for p in ((0, 1), (1, 0))]
    p = (0, 1) if v[0] >= v[1] else (1, 0)
    return p, max(v), [sisdr(refs[0], outs[p[0]]), sisdr(refs[1], outs[p[1]])]


def main(mix_dir, sep_root, audio_dir, out):
    mix_dir, sep_root, out = Path(mix_dir), Path(sep_root), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open(mix_dir / 'manifest.jsonl'))}
    geo = json.load(open(DB))['datasets']['cv_dev93']
    src_cache, rows, max_early_diff = {}, [], 0.0
    for key in sorted(man):
        m = man[key]; g = L[m['group']]; T = m['T_8k']
        mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float)    # (3, 6)
        cen = mic.mean(1); ref_az = math.degrees(math.atan2(mic[1, 0] - cen[1], mic[0, 0] - cen[0]))
        obs, _ = sf.read(mix_dir / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True); obs = obs.T
        refs = {}
        for k in 'AB':
            if g[k]['file'] not in src_cache:
                x, _ = sf.read(Path(audio_dir) / g[k]['file'], dtype='float64'); src_cache[g[k]['file']] = resample_poly(x, 1, 2)
            s = src_cache[g[k]['file']]; info = m['sources'][k]
            h, _ = sf.read(Path(g['rir']['rir_dir']) / f"h_{info['pos']}.wav", dtype='float64', always_2d=True); h = h.T
            rs = rir_start(h); assert rs == info['rir_start']
            he = h.copy(); he[:, rs + EARLY:] = 0
            off = g[k]['offset_8k'] - rs
            full = extract_piece(fftconvolve(s, h[0]), off, T) * info['gain']
            early_re = extract_piece(fftconvolve(s, he[0]), off, T) * info['gain']
            early_file, _ = sf.read(mix_dir / key / f'early_{k}_6ch_8k.wav', dtype='float64', always_2d=True)
            max_early_diff = max(max_early_diff, float(np.abs(early_re - early_file[:, 0]).max()))
            refs[k] = dict(early=early_file[:, 0], full=full)
        assert max_early_diff < 1e-6, (key, max_early_diff)
        t = m['target']; i = 'B' if t == 'A' else 'A'
        th1 = g['rir']['theta1']; th2 = g['rir']['theta2']
        oracle = {nm: wrap(doa(refs[k]['early'], obs, mic[:2].T) - ref_az) for nm, k in (('target', t), ('interferer', i))}
        base = dict(key=key, group=m['group'], swap=m['swap'], stratum=g['rir']['stratum'], t60=g['rir']['t60'],
                    overlap=m['overlap_active_frames'], sir_full=m['sir_mic0_full_db'], theta1=th1, theta2=th2,
                    oracle_doa_err_target=abs(wrap(oracle['target'] - th1)), oracle_doa_err_interferer=abs(wrap(oracle['interferer'] - th2)),
                    mix_sisdr={rt: [sisdr(refs[t][rt], obs[0]), sisdr(refs[i][rt], obs[0])] for rt in ('early', 'full')})
        for fe in FRONTENDS:
            y = np.stack([sf.read(sep_root / fe / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
            assert y.shape == (2, T)
            res = {}
            for rt in ('early', 'full'):
                p, avg, per = pit([refs[t][rt], refs[i][rt]], y)
                mx = base['mix_sisdr'][rt]
                res[rt] = dict(target_stream=p[0], sisdr_avg=avg, sisdr_target=per[0], sisdr_interferer=per[1],
                               sisdri_avg=avg - np.mean(mx), sisdri_target=per[0] - mx[0])
            az = [wrap(doa(y[k], obs, mic[:2].T) - ref_az) for k in range(2)]
            d = [abs(wrap(az[k] - th1)) for k in range(2)]
            dir_stream = 0 if d[0] <= d[1] else 1
            ts = res['early']['target_stream']
            rows.append(dict(base, frontend=fe, ref=res, doa=az, dir_stream=dir_stream, ref_stream=ts,
                             ref_full_agrees=res['full']['target_stream'] == ts, dir_correct=dir_stream == ts,
                             doa_err_target_stream=abs(wrap(az[ts] - th1)), doa_err_interferer_stream=abs(wrap(az[1 - ts] - th2)),
                             stream_rms=[float(np.sqrt(np.mean(v ** 2))) for v in y]))
        print('DONE', key, flush=True)

    def summ(sel):
        if not sel:
            return dict(n=0)
        q = lambda f: float(np.mean([f(r) for r in sel]))  # noqa: E731
        return dict(n=len(sel), dir_acc=q(lambda r: r['dir_correct']),
                    sisdr_avg_early=q(lambda r: r['ref']['early']['sisdr_avg']), sisdri_avg_early=q(lambda r: r['ref']['early']['sisdri_avg']),
                    sisdr_avg_full=q(lambda r: r['ref']['full']['sisdr_avg']), sisdri_avg_full=q(lambda r: r['ref']['full']['sisdri_avg']),
                    sisdr_target_early=q(lambda r: r['ref']['early']['sisdr_target']), sisdri_target_early=q(lambda r: r['ref']['early']['sisdri_target']),
                    sisdr_target_full=q(lambda r: r['ref']['full']['sisdr_target']), sisdri_target_full=q(lambda r: r['ref']['full']['sisdri_target']),
                    median_doa_err_target_stream=float(np.median([r['doa_err_target_stream'] for r in sel])),
                    median_doa_err_interferer_stream=float(np.median([r['doa_err_interferer_stream'] for r in sel])),
                    ref_full_agrees=q(lambda r: r['ref_full_agrees']))
    rep = dict(spec='docs/frontend_mix_v1_plan.md', max_abs_diff_recomputed_early_vs_file=max_early_diff, threshold=THRESH, frontends={})
    base_rows = [r for r in rows if r['frontend'] == FRONTENDS[0]]
    rep['oracle_doa_check'] = dict(
        median_err_target=float(np.median([r['oracle_doa_err_target'] for r in base_rows])),
        median_err_interferer=float(np.median([r['oracle_doa_err_interferer'] for r in base_rows])),
        frac_err_le_10deg=float(np.mean([e <= 10 for r in base_rows for e in (r['oracle_doa_err_target'], r['oracle_doa_err_interferer'])])),
        mixture_sisdr_early_mean=float(np.mean([np.mean(r['mix_sisdr']['early']) for r in base_rows])),
        mixture_sisdr_full_mean=float(np.mean([np.mean(r['mix_sisdr']['full']) for r in base_rows])))
    for fe in FRONTENDS:
        R = [r for r in rows if r['frontend'] == fe]
        gate_rows = [r for r in R if r['stratum'] != '[0,15)']
        acc = float(np.mean([r['dir_correct'] for r in gate_rows]))
        strata = {'stratum': {s: summ([r for r in R if r['stratum'] == s]) for s in sorted({r['stratum'] for r in R})}}
        for k, bins in BINS.items():
            strata[k] = {f'[{lo:g},{hi:g})': summ([r for r in R if lo <= r[k] < hi]) for lo, hi in bins}
        rms = np.array([r['stream_rms'] for r in R])
        rep['frontends'][fe] = dict(
            all=summ(R), gate_samples=len(gate_rows), dir_acc_excl_0_15=acc,
            decision='C5 allowed' if acc >= THRESH - 1e-12 else 'not C5 (C4 only)',
            failed_samples_excl_0_15=[dict(key=r['key'], stratum=r['stratum'], theta1=r['theta1'], theta2=r['theta2'], doa=r['doa'],
                                           dir_stream=r['dir_stream'], ref_stream=r['ref_stream']) for r in gate_rows if not r['dir_correct']],
            stream_rms_ratio_median=float(np.median(rms.min(1) / np.maximum(rms.max(1), 1e-20))),
            near_silent_stream_samples=int((rms.min(1) < 1e-4 * rms.max(1)).sum()), strata=strata)
    (out / 'eval_frontend_mix_v1.json').write_text(json.dumps(rep, indent=1) + '\n')
    with open(out / 'per_sample.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'frontends'}, indent=1))
    for fe, v in rep['frontends'].items():
        print(fe, json.dumps({k: v[k] for k in ('all', 'gate_samples', 'dir_acc_excl_0_15', 'decision', 'stream_rms_ratio_median',
                                                 'near_silent_stream_samples')}))


if __name__ == '__main__':
    main(*sys.argv[1:5])
