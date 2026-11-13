"""Front-end sanity checks A/B/C (docs/frontend_sanity_v1.md; rules frozen before running). Writes metrics JSON only.
A: SMS-WSJ test_eval92 first 50 (original dataset, direct-path references, original scores()) vs test_utterances.jsonl.
B: cv_dev93 50 groups (random.Random(2026).sample): mix_slurp_v1 flow on WSJ sources (in memory) vs official observation.
C: same 50 official mixtures truncated to mix_v1 durations (max two-speaker active-frame overlap window).
Inference lines identical to scripts/sms_wsj/frontend_mix_v1.py. Read-only w.r.t. ../speech_separation and data/.
Usage: python -B scripts/sms_wsj/frontend_sanity_v1.py --model M --out DIR [--n-test 50] [--n-cv 50] [--cpu]
"""
import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
MC = Path('<SEPARATION_REPO>/multichannel')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend_mix_v1 import RUNS, sha  # noqa: E402
from mix_slurp_v1 import EARLY, LEVEL_DBFS, active_power, extract_piece, frame_activity, rir_start  # noqa: E402

DB = '<SEPARATION_DATA_ROOT>/data/sms_wsj/sms_wsj.json'
DIRECT = MC / 'data/sms_wsj_direct_v1'
MIX_LIST = PROJ / 'configs/mix_v1_devel100_v2.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(RUNS), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-test', type=int, default=50); ap.add_argument('--n-cv', type=int, default=50); ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    run = RUNS[a.model]
    sys.path.insert(0, str(MC / a.model))
    import torch
    import train as T  # original code, unmodified
    from tfgridnet.objectives import sisdr as tsisdr
    rec = json.loads((run / 'test_summary.json').read_text()); cfg = json.loads((run / 'run_config.json').read_text())
    pkg = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    prov = dict(sha_matches_recorded=sha(run / 'best.pt') == rec['checkpoint_sha256'], config_equals_run_config=pkg['config'] == cfg)
    assert all(prov.values()), prov
    dev = 'cpu' if a.cpu else 'cuda'
    torch.set_num_threads(4); T.seed_all()
    model = T.Model(6).to(dev); model.load_state_dict(pkg['model']); model.eval()

    def infer(x):  # identical to frontend_mix_v1.py
        mix = torch.from_numpy(x)[None].to(dev)
        xin, _, scale = T.normalize(mix, mix[..., :2].transpose(1, 2))
        return (model(xin) * scale)[0].cpu().numpy(), mix

    S = lambda r, e: float(tsisdr(torch.from_numpy(np.asarray(r, np.float64)), torch.from_numpy(np.asarray(e, np.float64))))  # noqa: E731

    def sdri(refs, outs, mic0):
        v = [np.mean([S(refs[0], outs[p[0]]), S(refs[1], outs[p[1]])]) for p in ((0, 1), (1, 0))]
        base = np.mean([S(refs[0], mic0), S(refs[1], mic0)])
        return dict(sisdr=float(max(v)), sisdri=float(max(v) - base), mix_sisdr=float(base), perm=int(np.argmax(v)))

    a.out.mkdir(parents=True, exist_ok=True)
    rep = dict(model=a.model, device=dev, provenance=prov)
    t0 = time.perf_counter()
    # ---------------- A
    ds = T.SMSWSJ(Path(cfg['data_root']), 'test_eval92', tuple(range(6)), segment=None)
    old = {r['id']: r for r in map(json.loads, (run / 'test_utterances.jsonl').read_text().splitlines())}
    A = []
    with torch.no_grad():
        for i in range(a.n_test):
            item = ds[i]; r = ds.records[i]
            x, sr = sf.read(r['mixture'], dtype='float32', always_2d=True)
            same = bool(sr == 8000 and np.array_equal(x, item['mixture'].numpy()))
            pred, mix = infer(x)
            row = T.scores(item['target'][None].to(dev), torch.from_numpy(pred)[None].to(dev), mix, torch.tensor([item['length']], device=dev))[0]
            o = old[item['id']]
            A.append(dict(id=item['id'], file_equals_loader=same, si_sdr_db=row['si_sdr_db'], old_si_sdr_db=o['si_sdr_db'],
                          d_si_sdr=row['si_sdr_db'] - o['si_sdr_db'], d_si_sdri=row['si_sdri_db'] - o['si_sdri_db'],
                          perm_same=row['permutation'] == o['permutation']))
    dA = np.abs([x['d_si_sdr'] for x in A])
    rep['A'] = dict(n=len(A), all_files_equal_loader=all(x['file_equals_loader'] for x in A), max_abs_diff_db=float(dA.max()),
                    mean_abs_diff_db=float(dA.mean()), perm_mismatches=sum(not x['perm_same'] for x in A),
                    mean_si_sdr_new=float(np.mean([x['si_sdr_db'] for x in A])), mean_si_sdr_old=float(np.mean([x['old_si_sdr_db'] for x in A])),
                    problem=bool(dA.max() > 0.05 or dA.mean() > 0.01 or any(not x['perm_same'] for x in A)), rows=A)
    print('A', json.dumps({k: v for k, v in rep['A'].items() if k != 'rows'}), flush=True)
    # ---------------- B and C
    db = json.load(open(DB))['datasets']['cv_dev93']
    ids = random.Random(2026).sample(sorted(db), a.n_cv)
    Ls = [g['T_8k'] for g in json.load(open(MIX_LIST))['groups']]; Ls = Ls + Ls   # 200 sample lengths (2 swaps per group)
    rng = np.random.default_rng(2026); Lpick = rng.choice(Ls, size=len(ids))
    B, C = [], []
    with torch.no_grad():
        for n, eid in enumerate(ids):
            e = db[eid]; ap_ = e['audio_path']; Tn = e['num_samples']['observation']
            src = [sf.read(p, dtype='float64')[0] for p in ap_['original_source']]
            h = [sf.read(p, dtype='float64', always_2d=True)[0].T for p in ap_['rir']]
            full, early, gains = [], [], []
            for k in range(2):
                rs = rir_start(h[k]); he = h[k].copy(); he[:, rs + EARLY:] = 0
                off = e['offset'][k] - rs
                f_ = extract_piece(fftconvolve(src[k][None], h[k], axes=-1), off, Tn)
                e_ = extract_piece(fftconvolve(src[k][None], he, axes=-1), off, Tn)
                g = 10 ** (LEVEL_DBFS / 20) / np.sqrt(active_power(e_[0])[0])
                full.append(f_ * g); early.append(e_ * g); gains.append(g)
            clean = full[0] + full[1]
            noise = np.random.RandomState(2026 + n).normal(size=clean.shape)
            noise *= np.sqrt(np.mean(clean ** 2) / 10 ** (e['snr'] / 10) / np.mean(noise ** 2))
            ours = (clean + noise).T.astype(np.float32)
            p_ours, _ = infer(ours)
            r_ours = sdri([early[0][0], early[1][0]], p_ours, ours[:, 0].astype(np.float64))
            obs, _ = sf.read(ap_['observation'], dtype='float32', always_2d=True)
            p_off, _ = infer(obs)
            ref_e = [sf.read(p, dtype='float64', always_2d=True)[0][:, 0] for p in ap_['speech_reverberation_early']]
            ref_d = [sf.read(DIRECT / 'cv_dev93' / f'{eid}_{k}.wav', dtype='float64', always_2d=True)[0][:, 0] for k in range(2)]
            o64 = obs[:, 0].astype(np.float64)
            r_off_e, r_off_d = sdri(ref_e, p_off, o64), sdri(ref_d, p_off, o64)
            B.append(dict(id=eid, T=Tn, log_weights=e['log_weights'], ours_early=r_ours, official_early=r_off_e, official_direct=r_off_d))
            # C: truncation
            L = int(min(Lpick[n], Tn)); truncated = L < Tn
            _, a0 = frame_activity(ref_e[0]); _, a1 = frame_activity(ref_e[1])
            both = np.repeat(a0 & a1, 256)[:Tn]; both = np.pad(both, (0, Tn - len(both)))
            cs = np.concatenate([[0], np.cumsum(both)])
            starts = np.arange(0, Tn - L + 1, 800)
            st = int(starts[np.argmax(cs[starts + L] - cs[starts])])
            w = slice(st, st + L)
            p_tr, _ = infer(np.ascontiguousarray(obs[w]))
            r_tr_e = sdri([x[w] for x in ref_e], p_tr, o64[w]); r_tr_d = sdri([x[w] for x in ref_d], p_tr, o64[w])
            C.append(dict(id=eid, T=Tn, L=L, truncated=truncated, start=st, overlap_frac_window=float(cs[st + L] - cs[st]) / L,
                          full_early=r_off_e, trunc_early=r_tr_e, full_direct=r_off_d, trunc_direct=r_tr_d))
            print('BC', n, eid, round(r_ours['sisdri'], 2), round(r_off_e['sisdri'], 2), round(r_tr_e['sisdri'], 2), flush=True)
    m = lambda rows, k1, k2='sisdri': float(np.mean([r[k1][k2] for r in rows]))  # noqa: E731
    rep['B'] = dict(n=len(B), ours_early_sisdri=m(B, 'ours_early'), official_early_sisdri=m(B, 'official_early'),
                    official_direct_sisdri=m(B, 'official_direct'), ours_mix_sisdr=m(B, 'ours_early', 'mix_sisdr'),
                    official_mix_sisdr=m(B, 'official_early', 'mix_sisdr'),
                    log_weight_abs_diff_db=dict(median=float(np.median([abs(r['log_weights'][0] - r['log_weights'][1]) for r in B])),
                                                max=float(max(abs(r['log_weights'][0] - r['log_weights'][1]) for r in B))), rows=B)
    rep['B']['diff_db'] = rep['B']['ours_early_sisdri'] - rep['B']['official_early_sisdri']
    rep['B']['problem'] = bool(abs(rep['B']['diff_db']) > 2.0)
    rep['C'] = dict(n=len(C), truncated=sum(r['truncated'] for r in C), L_seconds=dict(min=min(r['L'] for r in C) / 8000, max=max(r['L'] for r in C) / 8000),
                    full_early_sisdri=m(C, 'full_early'), trunc_early_sisdri=m(C, 'trunc_early'),
                    full_direct_sisdri=m(C, 'full_direct'), trunc_direct_sisdri=m(C, 'trunc_direct'), rows=C)
    rep['seconds'] = time.perf_counter() - t0
    (a.out / f'sanity_{a.model}.json').write_text(json.dumps(rep, indent=1) + '\n')
    print('SUMMARY', json.dumps({k: ({q: w for q, w in v.items() if q != 'rows'} if isinstance(v, dict) else v) for k, v in rep.items()}), flush=True)


if __name__ == '__main__':
    main()
