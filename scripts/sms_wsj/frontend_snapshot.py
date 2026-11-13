"""Snapshot front-end diagnosis (docs/frontend_compare_v1.md, approved). IN-TRAINING snapshot, diagnosis only; no test used.
Loads the ORIGINAL train.py of the model (Model, normalize, seed_all) and the COPIED best.pt (sha256 checked against
snapshot.json; config equal to run_config.json after a JSON round trip). Inference identical to frontend_mix_v1.py.
  --mode cv  : cv_dev93 50 groups (random.Random(2026).sample), in memory; official direct references; PIT SI-SDRi; stream
               polarity vs own direct reference; selection accuracy before/after the v3 polarity rule (doa / choose unchanged).
               Writes OUT/cv_check.json with PASS/FAIL: FAIL if any error, output shape != (2, T), or mean direct SI-SDRi <= 0.
  --mode mix : only if cv_check.json says PASS; mix_v1 200 obs -> SEP_OUT/gNNN_sS/out{0,1}.wav (refuses to overwrite).
Usage (tfgridnet env, GPU): python -B scripts/sms_wsj/frontend_snapshot.py --model {tfgridnet,spatialnet} --snapshot DIR --mode {cv,mix} --out DIR [--sep-out DIR]
"""
import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
MC = Path('<SEPARATION_REPO>/multichannel')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frontend_mix_v1 import DB, doa, pit, sisdr, wrap  # noqa: E402
from eval_frontend_mix_v1_v3 import choose, polarity_fix  # noqa: E402

RUNDIR = dict(tfgridnet=MC / 'tfgridnet/exp/sms_direct_6ch_v1', spatialnet=MC / 'spatialnet/exp/sms_spatialnet_small_v1')
DIRECT = MC / 'data/sms_wsj_direct_v1/cv_dev93'
GEOM = PROJ / 'results/sms_wsj_geometry/cv_dev93_per_mixture.csv'
sys.path.append(str(PROJ / 'scripts/qwen'))
import split_cfg  # noqa: E402  (2026-11-12 SLURP test: env SLURP_EVAL_SPLIT=test -> test mix; cv check stays cv_dev93)
MIX = split_cfg.MIX  # devel: <SHARED_ROOT>/data/mix_v1/devel_seed2026 (unchanged)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(RUNDIR), required=True); ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--mode', choices=('cv', 'mix'), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sep-out', type=Path); ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--final', action='store_true', help='Label a verified final checkpoint; inference rules unchanged')
    a = ap.parse_args()
    snap = json.load(open(a.snapshot / 'snapshot.json'))
    if a.final:
        assert snap['completed_epoch'] == 100 and snap['latest_complete'] is True
    sys.path.insert(0, str(MC / a.model))
    import torch
    import train as T  # original code, unmodified
    assert sha(a.snapshot / 'best.pt') == snap['sha256_copy']
    pkg = torch.load(a.snapshot / 'best.pt', map_location='cpu', weights_only=False)
    assert json.loads(json.dumps(pkg['config'])) == json.loads((RUNDIR[a.model] / 'run_config.json').read_text())
    dev = 'cpu' if a.cpu else 'cuda'
    torch.set_num_threads(4); T.seed_all()
    model = T.Model(6).to(dev); model.load_state_dict(pkg['model']); model.eval()
    info = dict(model=a.model, snapshot=str(a.snapshot), epoch=pkg['epoch'], sha256=snap['sha256_copy'], node=os.environ.get('SLURMD_NODENAME', socket.gethostname()),
                gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                note=('FINAL trained checkpoint; no test used in this evaluation' if a.final else
                      'IN-TRAINING SNAPSHOT, diagnosis only, not a finished model; no test used'))
    print('INFO', json.dumps(info), flush=True)

    def infer(x):  # identical to frontend_mix_v1.py
        mix = torch.from_numpy(x)[None].to(dev)
        xin, _, scale = T.normalize(mix, mix[..., :2].transpose(1, 2))
        t0 = time.perf_counter()
        y = (model(xin) * scale)[0].detach().cpu().numpy().astype(np.float64)
        if dev == 'cuda':
            torch.cuda.synchronize()
        return y, time.perf_counter() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'cv':
        db = json.load(open(DB))['datasets']['cv_dev93']; geo = {r['id']: r for r in csv.DictReader(open(GEOM))}
        ids = random.Random(2026).sample(sorted(db), 50); rows, errors = [], []
        with torch.no_grad():
            for eid in ids:
                try:
                    e = db[eid]; x, sr = sf.read(e['audio_path']['observation'], dtype='float32', always_2d=True); assert sr == 8000
                    y, sec = infer(x)
                    if y.shape != (2, len(x)):
                        errors.append(dict(id=eid, error=f'shape {y.shape}')); continue
                    obs = x.T.astype(np.float64)
                    D = [sf.read(DIRECT / f'{eid}_{k}.wav', dtype='float64', always_2d=True)[0][:, 0] for k in range(2)]
                    p, avg, per = pit(D, y)
                    mixs = float(np.mean([sisdr(D[k], obs[0]) for k in range(2)]))
                    corr = {p[k]: float(np.dot(y[p[k]], D[k]) / (np.linalg.norm(y[p[k]]) * np.linalg.norm(D[k]) + 1e-20)) for k in range(2)}
                    mic = np.asarray(e['sensor_position'], float); c = mic.mean(1); ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
                    th = [float(geo[eid]['az_rel_mic0_0']), float(geo[eid]['az_rel_mic0_1'])]
                    yf, flipped = polarity_fix(y, obs[0])
                    az0 = [wrap(doa(y[k], obs, mic[:2].T) - ra) for k in range(2)]; az1 = [wrap(doa(yf[k], obs, mic[:2].T) - ra) for k in range(2)]
                    sep = float(geo[eid]['separation'])
                    for tgt in range(2):
                        rows.append(dict(id=eid, target=tgt, small_angle=sep < 15, sisdri=avg - mixs, sisdr=avg, mix_sisdr=mixs, seconds=sec,
                                         corr_own=[corr[0], corr[1]], flipped=[bool(f) for f in flipped],
                                         correct_before=choose(az0, th[tgt]) == p[tgt], correct_after=choose(az1, th[tgt]) == p[tgt]))
                except Exception as ex:  # record and fail
                    errors.append(dict(id=eid, error=repr(ex)))
        pm = rows[::2]
        q = lambda sel, f: float(np.mean([f(r) for r in sel])) if sel else None  # noqa: E731
        G = [r for r in rows if not r['small_angle']]
        summ = dict(n_mixtures=len(pm), errors=errors, mean_direct_sisdri=q(pm, lambda r: r['sisdri']), mean_direct_sisdr=q(pm, lambda r: r['sisdr']),
                    mean_mixture_sisdr=q(pm, lambda r: r['mix_sisdr']), corr_negative_stream0=q(pm, lambda r: r['corr_own'][0] < 0),
                    corr_negative_stream1=q(pm, lambda r: r['corr_own'][1] < 0), median_corr_stream0=float(np.median([r['corr_own'][0] for r in pm])) if pm else None,
                    median_corr_stream1=float(np.median([r['corr_own'][1] for r in pm])) if pm else None,
                    flip_rate_stream0=q(pm, lambda r: r['flipped'][0]), flip_rate_stream1=q(pm, lambda r: r['flipped'][1]),
                    sel_acc_before=q(rows, lambda r: r['correct_before']), sel_acc_after=q(rows, lambda r: r['correct_after']),
                    sel_acc_after_excl_0_15=q(G, lambda r: r['correct_after']), mean_seconds=q(pm, lambda r: r['seconds']))
        summ['verdict'] = 'PASS' if (not errors and len(pm) == 50 and summ['mean_direct_sisdri'] > 0) else 'FAIL'
        (a.out / 'cv_check.json').write_text(json.dumps(dict(info=info, summary=summ, rows=rows), indent=1) + '\n')
        print('CV', json.dumps(summ), flush=True)
        return
    chk = json.load(open(a.out / 'cv_check.json'))
    assert chk['summary']['verdict'] == 'PASS', 'cv_dev93 check did not pass; mix_v1 not run'
    a.sep_out.mkdir(parents=True, exist_ok=False); rows = []
    with torch.no_grad():
        for d in sorted(p for p in MIX.iterdir() if p.is_dir()):
            x, sr = sf.read(d / 'obs_6ch_8k.wav', dtype='float32', always_2d=True); assert sr == 8000
            y, sec = infer(x); assert y.shape == (2, len(x)) and np.isfinite(y).all()
            o = a.sep_out / d.name; o.mkdir()
            for k in range(2):
                sf.write(o / f'out{k}.wav', y[k].astype(np.float32), 8000, subtype='FLOAT')
            rows.append(dict(key=d.name, seconds=sec, samples=len(x)))
    assert len(rows) == (200 if split_cfg.SPLIT == 'devel' else sum(1 for _ in open(split_cfg.MANIFEST)))
    with open(a.sep_out / 'manifest.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    (a.sep_out / 'summary.json').write_text(json.dumps(dict(info=info, n=len(rows), mean_seconds=float(np.mean([r['seconds'] for r in rows]))), indent=1) + '\n')
    print('MIX_DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
