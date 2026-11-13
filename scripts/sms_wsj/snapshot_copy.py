"""Copy in-training best.pt snapshots for diagnosis (docs/frontend_compare_v1.md, approved rules). READ-ONLY on the training
directories: stat best.pt and metrics.jsonl, wait >= 60 s, stat again (must be identical), copy to a NEW shared directory,
stat the source again (must still be identical), check copy size, sha256 the COPY, then load the COPY to record epoch /
best / last dev SI-SDR / complete flag and compare its config with run_config.json. Writes snapshot.json next to the copy.
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/snapshot_copy.py OUT_ROOT [--wait 65]
"""
import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

MC = Path('<SEPARATION_REPO>/multichannel')
RUNS = dict(tfgridnet=MC / 'tfgridnet/exp/sms_direct_6ch_v1', spatialnet=MC / 'spatialnet/exp/sms_spatialnet_small_v1')


def st(p):
    s = os.stat(p); return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.st_mtime)))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_root', type=Path); ap.add_argument('--wait', type=int, default=65)
    a = ap.parse_args(); assert a.wait >= 60
    import torch
    first = {m: dict(best=st(r / 'best.pt'), metrics=st(r / 'metrics.jsonl')) for m, r in RUNS.items()}
    print('STAT1', json.dumps(first), flush=True)
    time.sleep(a.wait)
    second = {m: dict(best=st(r / 'best.pt'), metrics=st(r / 'metrics.jsonl')) for m, r in RUNS.items()}
    print('STAT2', json.dumps(second), flush=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for m, r in RUNS.items():
        if first[m] != second[m]:
            print('SKIP_CHANGED', m, flush=True); continue
        tmp = a.out_root / f'{m}_tmp_{stamp}'; tmp.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(r / 'best.pt', tmp / 'best.pt')
        third = dict(best=st(r / 'best.pt'), metrics=st(r / 'metrics.jsonl'))
        ok = third == second[m] and os.path.getsize(tmp / 'best.pt') == second[m]['best']['size']
        if not ok:
            shutil.rmtree(tmp); print('COPY_INVALID', m, json.dumps(third), flush=True); continue
        pkg = torch.load(tmp / 'best.pt', map_location='cpu', weights_only=False)
        hist = pkg.get('history') or []
        last = hist[-1] if isinstance(hist, list) and hist else None
        cfg_equal = pkg['config'] == json.loads((r / 'run_config.json').read_text())
        dest = a.out_root / f"{m}_ep{pkg['epoch']}_{stamp}"; tmp.rename(dest)
        rec = dict(model=m, note='IN-TRAINING SNAPSHOT, diagnosis only, not a finished model; no test used',
                   source=str(r / 'best.pt'), copy=str(dest / 'best.pt'), source_stat_before=first[m], source_stat_wait=second[m],
                   source_stat_after=third, wait_seconds=a.wait, sha256_copy=sha(dest / 'best.pt'), epoch=pkg['epoch'], best=pkg.get('best'),
                   complete=pkg.get('complete'), config_equals_run_config=cfg_equal,
                   history_last=(last if isinstance(last, dict) else None), history_len=len(hist) if isinstance(hist, list) else None,
                   copied_at=stamp)
        (dest / 'snapshot.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
        print('SNAPSHOT', json.dumps({k: rec[k] for k in ('model', 'copy', 'epoch', 'best', 'complete', 'config_equals_run_config', 'sha256_copy')}, default=str), flush=True)
        if isinstance(last, dict):
            print('   history_last', json.dumps({k: last.get(k) for k in ('epoch', 'dev_si_sdr_db', 'valid_loss')}), flush=True)


if __name__ == '__main__':
    main()
