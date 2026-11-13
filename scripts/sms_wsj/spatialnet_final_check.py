"""Verify completion under lora_v2_runs §22; optionally copy best.pt to a NEW directory.
Read-only on the training run. Reads existing SMS-WSJ summary for provenance only;
does not evaluate or access test audio. No checkpoint flags are modified.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch

RUN = Path('<SEPARATION_REPO>/multichannel/spatialnet/exp/sms_spatialnet_small_v1')
LOGS = RUN.parents[2] / 'logs'
INITIAL = '21977586'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--copy-root', type=Path)
    a = ap.parse_args()
    files = ['best.pt', 'latest.pt', 'metrics.jsonl', 'run_config.json', 'test_summary.json', 'continuation_jobs.txt']
    before = {n: stat(RUN / n) for n in files}
    jobs = [INITIAL] + re.findall(r'Submitted batch job (\d+)', (RUN / 'continuation_jobs.txt').read_text())
    accounting = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-X', '-n', '-P', '-o', 'JobID,State,ExitCode'], text=True)
    states = {r[0]: r[1:3] for line in accounting.splitlines() if (r := line.split('|'))[0]}
    assert set(states) == set(jobs) and all(s == ['COMPLETED', '0:0'] for s in states.values()), states
    queued = subprocess.check_output(['squeue', '-u', '<USER>', '-h', '-o', '%i %j %T'], text=True)
    assert not any('sms-spatialnet-small' in line or line.split()[0] in jobs for line in queued.splitlines()), queued
    history = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in history] == list(range(1, 101))
    logged = {}
    for job in jobs:
        for line in (LOGS / f'spatialnet_train_{job}.out').read_text().splitlines():
            if line.startswith('EPOCH '):
                row = json.loads(line[6:]); logged[row['epoch']] = row
    assert all(logged.get(row['epoch']) == row for row in history)
    bestrow = min(history, key=lambda r: r['valid_loss'])
    cfg = json.loads((RUN / 'run_config.json').read_text())
    best = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=False)
    latest = torch.load(RUN / 'latest.pt', map_location='cpu', weights_only=False)
    assert latest['epoch'] == 100 and latest['complete'] is True and latest['history'] == history
    assert best['epoch'] == bestrow['epoch'] and best['best'] == bestrow['valid_loss']
    assert best['history'][-1] == bestrow
    assert json.loads(json.dumps(best['config'])) == json.loads(json.dumps(latest['config'])) == cfg
    summary = json.loads((RUN / 'test_summary.json').read_text())
    digest = sha(RUN / 'best.pt')
    assert summary['completed_epoch'] == 100 and summary['selected_epoch'] == best['epoch']
    assert summary['checkpoint_sha256'] == digest and summary['audit_sha256'] == cfg['audit_sha256']
    assert summary['smoke'] is False and summary['channels'] == cfg['channels']
    assert summary['target'] == cfg['target'] and summary['reference_channel'] == cfg['reference_channel']
    assert before == {n: stat(RUN / n) for n in files}
    record = dict(model='spatialnet', note='FINAL trained checkpoint; no new test evaluation',
                  source=str(RUN / 'best.pt'), source_stat_before=before, epoch=best['epoch'],
                  completed_epoch=100, complete=best['complete'], latest_complete=True,
                  best=best['best'], history_last=bestrow, config_equals_run_config=True,
                  sha256_copy=digest, jobs=states, summary_sha256=sha(RUN / 'test_summary.json'))
    if a.copy_root:
        dest = a.copy_root / f"spatialnet_ep{best['epoch']}_{datetime.now():%Y%m%d-%H%M%S}"
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(RUN / 'best.pt', dest / 'best.pt')
        assert stat(dest / 'best.pt')['bytes'] == before['best.pt']['bytes'] and sha(dest / 'best.pt') == digest
        assert before == {n: stat(RUN / n) for n in files}
        record.update(copy=str(dest / 'best.pt'), source_stat_after={n: stat(RUN / n) for n in files})
        (dest / 'snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
