"""Re-evaluate a finished SMS-WSJ separation checkpoint on test_eval92 and compare
with the originally recorded test metrics. Read-only with respect to ../speech_separation.

Supported runs (frozen, completed 100-epoch training):
  blstm_mvdr  -> multichannel/blstm_mvdr/exp/sms_blstm_mvdr_v1
  fasnet_tac  -> multichannel/fasnet_tac/exp/sms_fasnet_tac_v1

Design:
  * Imports the ORIGINAL train.py of that model and calls its own evaluate() unchanged,
    so data loading, normalisation, forward pass and scoring are exactly the recorded code.
  * evaluate() writes into its run_dir and hashes run_dir/best.pt, so we pass a fresh
    directory under ./outputs that contains only a symlink to the original best.pt.
    Nothing is written under ../speech_separation (bytecode writing is disabled too).
  * Before inference: checkpoint SHA256 == recorded test_summary, checkpoint config ==
    run_config.json, current source/data hashes == hashes stored in the checkpoint.
  * After inference: summary means and every per-utterance row are compared with the
    original test_utterances.jsonl; comparison.json records max absolute differences.

Usage:
  python -B scripts/sms_wsj/repro_sms_wsj_test.py --model blstm_mvdr --out outputs/repro/blstm_X
  python -B scripts/sms_wsj/repro_sms_wsj_test.py --model fasnet_tac --check-only   # CPU, no inference
"""
import sys
sys.dont_write_bytecode = True  # never write __pycache__ into the read-only source tree

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

MC = Path('<SEPARATION_REPO>/multichannel')
RUNS = {'blstm_mvdr': MC/'blstm_mvdr/exp/sms_blstm_mvdr_v1',
        'fasnet_tac': MC/'fasnet_tac/exp/sms_fasnet_tac_v1'}
METRICS = ('si_sdr_db', 'si_sdri_db', 'mixture_si_sdr_db')
# Abs. tolerance (dB) for calling the reproduction a match. Logged diffs are reported regardless.
TOL_MEAN, TOL_UTT = 1e-3, 1e-2


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(RUNS), required=True)
    ap.add_argument('--out', type=Path, help='new, empty output directory (inside this project)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check-only', action='store_true', help='provenance checks only, no GPU/inference')
    args = ap.parse_args()
    run = RUNS[args.model]

    # Import the original training module exactly as `python <model>/train.py` would see it.
    sys.path.insert(0, str(MC/args.model))
    import torch
    import train as T  # noqa: E402  (original code, unmodified)

    recorded = json.loads((run/'test_summary.json').read_text())
    run_config = json.loads((run/'run_config.json').read_text())
    ckpt_sha = sha(run/'best.pt')
    checks = {'checkpoint_sha256_matches_recorded': ckpt_sha == recorded['checkpoint_sha256']}

    package = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    checks['checkpoint_config_equals_run_config'] = package['config'] == run_config
    checks['selected_epoch_matches_recorded'] = package['epoch'] == recorded['selected_epoch']

    # Recompute current code/data provenance with the original configuration() function.
    data_root = Path(run_config['data_root'])
    tt = T.SMSWSJ(data_root, 'test_eval92', tuple(range(6)), segment=None)
    cfg_args = SimpleNamespace(channels=6, batch_size=2, epochs=100, smoke=False, data=data_root)
    current = T.configuration(cfg_args, None)
    checks['source_sha256_unchanged'] = current['source_sha256'] == package['config']['source_sha256']
    checks['audit_and_manifests_unchanged'] = (current['audit_sha256'] == package['config']['audit_sha256']
                                               and current['manifests'] == package['config']['manifests'])
    checks['test_utterances'] = len(tt)
    print('PROVENANCE ' + json.dumps(checks), flush=True)
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise SystemExit(f'Provenance check failed: {failed}')
    if args.check_only:
        print('CHECK_ONLY_PASS', flush=True)
        return 0

    assert args.out is not None, '--out is required for evaluation'
    assert torch.cuda.is_available(), 'Run through Slurm on a GPU node'
    out = args.out.resolve()
    assert Path.cwd().resolve() in out.parents, 'Output must be inside the project directory'
    out.mkdir(parents=True, exist_ok=False)
    (out/'best.pt').symlink_to(run/'best.pt')

    torch.set_num_threads(4)
    T.seed_all()                      # same determinism flags as the original run
    model = T.Model(6).cuda()
    model.load_state_dict(package['model'])
    gpu = torch.cuda.get_device_name()
    print(json.dumps(dict(gpu=gpu, torch=torch.__version__, model=args.model,
                          parameters=sum(p.numel() for p in model.parameters()))), flush=True)
    T.evaluate(model, tt, out, args.workers, smoke=False)   # original evaluation, writes into `out`

    new = json.loads((out/'test_summary.json').read_text())
    old_rows = {r['id']: r for r in map(json.loads, (run/'test_utterances.jsonl').read_text().splitlines())}
    new_rows = {r['id']: r for r in map(json.loads, (out/'test_utterances.jsonl').read_text().splitlines())}
    assert old_rows.keys() == new_rows.keys()
    report = dict(model=args.model, run_dir=str(run), gpu=gpu, torch=torch.__version__,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'), provenance=checks,
                  recorded={k: recorded[k] for k in METRICS}, reproduced={k: new[k] for k in METRICS},
                  mean_abs_diff={k: abs(new[k]-recorded[k]) for k in METRICS},
                  utt_max_abs_diff={k: max(abs(new_rows[i][k]-old_rows[i][k]) for i in old_rows) for k in METRICS},
                  permutation_mismatches=sum(new_rows[i]['permutation'] != old_rows[i]['permutation'] for i in old_rows),
                  utterances=len(new_rows), elapsed_seconds=new['elapsed_seconds'])
    report['match'] = (all(v <= TOL_MEAN for v in report['mean_abs_diff'].values())
                       and all(v <= TOL_UTT for v in report['utt_max_abs_diff'].values())
                       and report['permutation_mismatches'] == 0)
    report['tolerance_db'] = dict(mean=TOL_MEAN, utterance=TOL_UTT)
    (out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print('REPRO ' + json.dumps(report), flush=True)
    return 0 if report['match'] else 3


if __name__ == '__main__':
    sys.exit(main())
