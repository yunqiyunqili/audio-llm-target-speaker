"""Zero-shot front-end inference on mix_v1 (docs/frontend_mix_v1_plan.md). Read-only w.r.t. ../speech_separation.
Imports the ORIGINAL train.py of the model (Model, normalize, seed_all) unchanged; checkpoint sha256 must equal the
recorded test_summary and package config must equal run_config.json. Inference exactly as the original evaluate():
x = obs / std(obs); pred = model(x) * std; FP32, batch 1. Output: two mic-0 streams (2, T) per sample.
Writes OUT_ROOT/<model>/gNNN_sS/out{0,1}.wav (8 kHz, float32) + manifest.jsonl; refuses to overwrite.
--dry-run N: run N samples on the chosen device and write NOTHING to OUT_ROOT (prints shapes, time, size estimate).
Usage: python -B scripts/sms_wsj/frontend_mix_v1.py --model {blstm_mvdr,fasnet_tac} --mix-dir DIR --out-root DIR [--dry-run N] [--cpu]
"""
import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

MC = Path('<SEPARATION_REPO>/multichannel')
RUNS = {'blstm_mvdr': MC / 'blstm_mvdr/exp/sms_blstm_mvdr_v1', 'fasnet_tac': MC / 'fasnet_tac/exp/sms_fasnet_tac_v1'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(RUNS), required=True)
    ap.add_argument('--mix-dir', type=Path, required=True); ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--dry-run', type=int, default=0); ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    run = RUNS[a.model]
    sys.path.insert(0, str(MC / a.model))
    import torch
    import train as T  # original code, unmodified
    rec = json.loads((run / 'test_summary.json').read_text()); cfg = json.loads((run / 'run_config.json').read_text())
    ck = sha(run / 'best.pt')
    pkg = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    checks = dict(checkpoint_sha256=ck, sha_matches_recorded=ck == rec['checkpoint_sha256'],
                  config_equals_run_config=pkg['config'] == cfg, epoch=pkg['epoch'])
    print('PROVENANCE', json.dumps(checks), flush=True)
    assert checks['sha_matches_recorded'] and checks['config_equals_run_config']
    dev = 'cpu' if a.cpu else 'cuda'
    if dev == 'cuda':
        assert torch.cuda.is_available()
    torch.set_num_threads(4); T.seed_all()
    model = T.Model(6).to(dev); model.load_state_dict(pkg['model']); model.eval()
    samples = sorted(p.name for p in a.mix_dir.iterdir() if p.is_dir() and (p / 'obs_6ch_8k.wav').exists())
    assert len(samples) == 200, len(samples)
    out = a.out_root / a.model
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=False)
    todo = samples[:a.dry_run] if a.dry_run else samples
    rows, nbytes, t0 = [], 0, time.perf_counter()
    with torch.no_grad():
        for key in todo:
            x, sr = sf.read(a.mix_dir / key / 'obs_6ch_8k.wav', dtype='float32', always_2d=True)
            assert sr == 8000 and x.shape[1] == 6
            mix = torch.from_numpy(x)[None].to(dev)                  # (1, T, 6) as in the shared loader
            xin, _, scale = T.normalize(mix, mix[..., :2].transpose(1, 2))
            t1 = time.perf_counter()
            pred = (model(xin) * scale)[0].cpu().numpy()               # (2, T)
            sec = time.perf_counter() - t1
            assert pred.shape == (2, len(x)) and np.isfinite(pred).all(), pred.shape
            r = dict(key=key, samples=len(x), seconds=sec, peak=[float(np.abs(p).max()) for p in pred],
                     rms=[float(np.sqrt(np.mean(p ** 2))) for p in pred])
            nbytes += 2 * (len(x) * 4 + 44)
            if not a.dry_run:
                d = out / key; d.mkdir()
                for k in range(2):
                    sf.write(d / f'out{k}.wav', pred[k], 8000, subtype='FLOAT')
                r['dir'] = str(d)
            rows.append(r)
    tot = time.perf_counter() - t0
    est_total = nbytes / len(todo) * len(samples)
    summ = dict(model=a.model, device=dev, samples=len(todo), seconds=tot, bytes=nbytes, estimated_bytes_all_200=est_total,
                provenance=checks, torch=torch.__version__, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if not a.dry_run:
        with open(out / 'manifest.jsonl', 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        (out / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    print('SUMMARY', json.dumps(summ), flush=True)
    for r in rows[:3]:
        print('ROW', json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
