"""S0 devel equivalence check for the SLURP-test branches (docs/slurp_test_freeze_v1.md; 2026-11-12). CPU only.

Re-runs every modified construction / selection / scoring script with the DEVEL default (SLURP_EVAL_SPLIT unset) into a fresh
directory and compares with the recorded devel outputs: byte-identical files, or identical JSON after removing fields that only
hold run-specific paths or timings (listed per check). Inference scripts are checked with their CPU --dry-run: same items and
the same prompt text / processed-audio sha256 as the recorded devel runs. Any mismatch -> exit 1. Never touches test data.
Steps (one CPU job each; same commit):  construct | select | direct | score | dryrun
Usage: python -I -B scripts/lora/slurp_test_s0_check.py STEP OUT_DIR
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
SH = Path('<SHARED_ROOT>')
PY = dict(qwen_omni=str(SH / 'envs/qwen_omni/bin/python'), qwen_omni_lora=str(SH / 'envs/qwen_omni_lora/bin/python'),
          tfgridnet='<CONDA_ROOT>/envs/tfgridnet/bin/python', sms_wsj='<SEPARATION_REPO>/envs/sms_wsj/bin/python')
AUDIO = SH / 'data/slurp/audio/slurp_real'
MIXD = SH / 'data/mix_v1/devel_seed2026'
SEP = SH / 'data/mix_v1/separated_final/spatialnet_ep100_20261107-223434'
RES = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(env, *args, flags=('-I', '-B')):
    e = {k: v for k, v in os.environ.items() if k != 'SLURP_EVAL_SPLIT'}
    e.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HOME=str(SH / 'checkpoints/hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    cmd = [PY[env], *flags, *map(str, args)]
    print('RUN', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=PROJ, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:], p.stderr[-4000:], flush=True); raise SystemExit(f'COMMAND_FAILED {cmd}')
    return p.stdout


def ok(name, cond, detail=None):
    RES[name] = dict(passed=bool(cond), **({'detail': detail} if detail is not None else {}))
    print('CHECK', 'PASS' if cond else 'FAIL', name, '' if detail is None else json.dumps(detail)[:400], flush=True)


def same_bytes(name, a, b):
    ok(name, sha(a) == sha(b), dict(new=str(a), recorded=str(b)))


def strip(o, drop):
    if isinstance(o, dict):
        return {k: strip(v, drop) for k, v in o.items() if k not in drop}
    if isinstance(o, list):
        return [strip(v, drop) for v in o]
    return o


def same_json(name, a, b, drop=(), lines=False):
    ld = (lambda p: [json.loads(x) for x in open(p)]) if lines else (lambda p: json.load(open(p)))
    ok(name, strip(ld(a), set(drop)) == strip(ld(b), set(drop)), dict(new=str(a), recorded=str(b), ignored=list(drop)))


def construct(out):
    run('qwen_omni', 'scripts/slurp/make_mix_list_v1.py', 'data/slurp/manifests/split_v2_seed2026', AUDIO, out / 'list.json', '--v2')
    same_bytes('mix_list_v2', out / 'list.json', PROJ / 'configs/mix_v1_devel100_v2.json')
    run('qwen_omni', 'scripts/sms_wsj/mix_slurp_v1.py', out / 'list.json', AUDIO, out / 'mix')
    rec = sorted(p.relative_to(MIXD) for p in MIXD.rglob('*.wav')); new = sorted(p.relative_to(out / 'mix') for p in (out / 'mix').rglob('*.wav'))
    bad = [str(r) for r in rec if sha(MIXD / r) != sha(out / 'mix' / r)] if rec == new else ['file sets differ']
    ok('mix_audio_bytes', not bad and len(rec) == 800, dict(n_files=len(rec), mismatches=bad[:5]))
    same_json('mix_manifest', out / 'mix/manifest.jsonl', MIXD / 'manifest.jsonl', drop=('dir',), lines=True)
    same_json('mix_summary', out / 'mix/manifest_summary.json', MIXD / 'manifest_summary.json', drop=('seconds',))
    REV = SH / 'data/lora_v1/devel_rev_cv_dev93'
    run('qwen_omni', 'scripts/lora/reverb_single.py', '--split', 'devel', '--out', out / 'rev')
    rec = sorted(p.name for p in REV.glob('*.wav')); new = sorted(p.name for p in (out / 'rev').glob('*.wav'))
    bad = [r for r in rec if sha(REV / r) != sha(out / 'rev' / r)] if rec == new else ['file sets differ']
    ok('c1r_audio_bytes', not bad and len(rec) == 428, dict(n_files=len(rec), mismatches=bad[:5]))
    old = [json.loads(x) for p in sorted(REV.glob('manifest_part*.jsonl')) for x in open(p)]
    newm = [json.loads(x) for x in open(out / 'rev/manifest_part0.jsonl')]
    ok('c1r_manifest', sorted(old, key=lambda r: r['file']) == newm, dict(n=len(newm)))
    for d in ('mix', 'rev'):  # large audio is not kept (hashes recorded above)
        for p in (out / d).rglob('*.wav'):
            p.unlink()


def select(out):
    run('tfgridnet', 'scripts/sms_wsj/select_final.py', SEP, out / 'sel', flags=('-B',))
    R = PROJ / 'outputs/frontend_final_eval/ep100_20261107'
    for f in ('spatialnet/selection.jsonl', 'spatialnet/per_sample.jsonl', 'eval_snapshot_mix.json'):
        same_bytes('select_' + f, out / 'sel' / f, R / f)
    same_json('select_other_selection', out / 'sel/spatialnet/other_selection.jsonl', R / 'spatialnet/other_selection.jsonl', drop=('source_selection',), lines=True)


def direct(out):
    import numpy as np
    run('sms_wsj', 'scripts/sms_wsj/direct_refs_mix_v1.py', MIXD, AUDIO, out / 'dr', flags=('-B',))
    R = PROJ / 'outputs/frontend_mix_v1_eval_v2'
    same_json('direct_refs_summary', out / 'dr/direct_refs_summary.json', R / 'direct_refs_summary.json')
    bad = []
    for p in sorted((R / 'direct_refs').glob('*.npz')):
        a, b = np.load(p), np.load(out / 'dr/direct_refs' / p.name)
        if sorted(a.files) != sorted(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            bad.append(p.name)
    ok('direct_refs_arrays', not bad and len(list((R / 'direct_refs').glob('*.npz'))) == 200, dict(mismatches=bad[:5]))


def score(out):
    o = out.relative_to(PROJ)
    sel = 'outputs/frontend_final_eval/ep100_20261107/spatialnet'
    run('qwen_omni', 'scripts/qwen/eval_final_routes.py', '--selected', 'outputs/qwen_omni_final/22029639_selected', '--other', 'outputs/qwen_omni_final/22029640_other',
        '--selection-dir', sel, '--c5', o / 'c5_base/eval.json', '--delta', o / 'delta_base', '--real', o / 'real_base')
    same_bytes('final_routes_base_c5', out / 'c5_base/eval.json', PROJ / 'outputs/c5_final/ep100_20261107/eval.json')
    same_bytes('final_routes_base_delta_for_plot.json', out / 'delta_base/for_plot.json', PROJ / 'outputs/c6_delta_final/ep100_20261107/for_plot.json')
    # S0 item 3 (accepted 2026-11-12): the 11-07 record predates the top-level 'route' key added in 4580ad6; all else must be equal
    a = json.load(open(out / 'delta_base/eval_c6_delta_spatialnet.json'))
    ok('final_routes_base_delta_eval_c6_delta_spatialnet_except_route_key', a.pop('route', None) == 'both' and a == json.load(open(PROJ / 'outputs/c6_delta_final/ep100_20261107/eval_c6_delta_spatialnet.json')))
    for f in ('eval_c6_real.json', 'risk_coverage.json'):
        same_bytes('final_routes_base_real_' + f, out / 'real_base' / f, PROJ / 'outputs/c6_real_final/ep100_20261107' / f)
    run('qwen_omni', 'scripts/qwen/eval_final_routes.py', '--lora', '--selected', 'outputs/lora_audio_final/22029676_selected', '--other', 'outputs/lora_audio_final/22029677_other',
        '--selection-dir', sel, '--c5', o / 'c5_lora/eval.json', '--delta', 'outputs/c6_delta_final/ep100_20261107', '--real', o / 'real_lora')
    same_bytes('final_routes_lora_c5', out / 'c5_lora/eval.json', PROJ / 'outputs/c5_final/lora_step440_20261107/eval.json')
    for f in ('eval_c6_real.json', 'risk_coverage.json'):
        same_bytes('final_routes_lora_real_' + f, out / 'real_lora' / f, PROJ / 'outputs/c6_real_final/lora_step440_20261107' / f)
    br, lr = 'outputs/c6_real_final/ep100_20261107/eval_c6_real.json', 'outputs/c6_real_final/lora_step440_20261107/eval_c6_real.json'
    t2s, t2o = 'outputs/lora_step2_eval/lr1e-4/step660/final_selected_all_22053124', 'outputs/lora_step2_eval/lr1e-4/step660/final_other_all_22053125'
    for kind, b, l, rec in (('p2', ('outputs/qwen_omni_final/22029639_selected', 'outputs/qwen_omni_final/22029640_other'),
                             ('outputs/lora_audio_final/22029676_selected', 'outputs/lora_audio_final/22029677_other'), 'outputs/lora_step1_checks/final_spatialnet_ep100_20261107'),
                            ('t2', ('outputs/qwen_omni_final/22029639_selected', 'outputs/qwen_omni_final/22029640_other'), (t2s, t2o),
                             'outputs/lora_step2_eval/final_checks_22053124_25_v3'),
                            ('audio_vs_t2', ('outputs/lora_audio_final/22029676_selected', 'outputs/lora_audio_final/22029677_other'), (t2s, t2o),
                             'outputs/lora_step2_eval/audio_vs_t2_final_checks_corrected')):
        args = ['--baseline-selected', b[0], '--baseline-other', b[1], '--lora-selected', l[0], '--lora-other', l[1], '--selection-dir', sel,
                '--baseline-real', br, '--lora-real', lr, '--route-kind', kind]
        run('qwen_omni', 'scripts/lora/eval_final_lora_checks.py', *args, '--out', o / f'checks_{kind}')
        if kind == 'p2':  # S0 item 4 (decision 2026-11-12): only metadata keys route_kind (4580ad6) and skipped_queries may be added
            a = json.load(open(out / f'checks_{kind}/checks.json')); b = {k: v for k, v in a.items() if k not in ('route_kind', 'skipped_queries')}
            ok('final_lora_checks_p2_values_and_bytes', json.dumps(b, indent=2, ensure_ascii=False) + '\n' == open(PROJ / rec / 'checks.json').read(),
               dict(added=sorted(set(a) - set(b)), skipped=a.get('skipped_queries')))
        else:
            same_bytes(f'final_lora_checks_{kind}', out / f'checks_{kind}/checks.json', PROJ / rec / 'checks.json')
        if kind == 'audio_vs_t2':  # --ci9833 adds keys only; everything else identical
            run('qwen_omni', 'scripts/lora/eval_final_lora_checks.py', *args, '--ci9833', '--out', o / f'checks_{kind}_ci9833')
            a = json.load(open(out / f'checks_{kind}_ci9833/checks.json')); r = json.load(open(PROJ / rec / 'checks.json'))
            ok('final_lora_checks_ci9833_only_adds', strip(a, {'ci9833'}) == r and 'ci9833' in a['conditions']['C5']['none']['all']['paired']['target_any']['eq'],
               a['conditions']['C5']['none']['all']['paired']['target_any']['eq'])
    # merge_test_parts round trip: split recorded devel C5 dirs into two parts and merge back -> identical raw files
    for src in ('outputs/qwen_omni_final/22029639_selected', 'outputs/lora_audio_final/22029676_selected'):
        cfg = json.load(open(PROJ / src / 'config.json')); conds = sorted(d.name for d in (PROJ / src).iterdir() if d.is_dir())
        parts = []
        for k in (0, 1):
            pd = out / 'merge_parts' / Path(src).name / f'part{k}'; pd.mkdir(parents=True)
            keys = sorted(cfg['audio_sha256']); half = keys[k * len(keys) // 2:(k + 1) * len(keys) // 2]
            (pd / 'config.json').write_text(json.dumps(dict(cfg, part=k, nparts=2, audio_sha256={x: cfg['audio_sha256'][x] for x in half})) + '\n')
            for c in conds:
                (pd / c).mkdir(); lines = (PROJ / src / c / 'raw.jsonl').read_text().splitlines()
                (pd / c / 'raw.jsonl').write_text(''.join(l + '\n' for l in lines if json.loads(l)['file'] in set(half)))
            parts.append(pd)
        mo = out / 'merged' / Path(src).name
        run('qwen_omni_lora', 'scripts/lora/merge_test_parts.py', mo, *parts)
        m = json.load(open(mo / 'config.json'))
        ok('merge_roundtrip_' + Path(src).name, all(sha(mo / c / 'raw.jsonl') == sha(PROJ / src / c / 'raw.jsonl') for c in conds) and m['audio_sha256'] == cfg['audio_sha256'], conds)
    run('qwen_omni_lora', 'scripts/lora/score_slurp_test_v1.py', '--devel-check', o / 'score_devel_check.json')
    ok('score_slurp_test_v1_devel_check', json.load(open(out / 'score_devel_check.json'))['C3_all'] == 'reproduced')
    # smoke run of the test report path on devel inputs (structure only; NOT a test result)
    g = lambda pat: sorted(str(p.relative_to(PROJ)) for p in PROJ.glob(pat))  # noqa: E731
    AE = 'outputs/lora_audio_eval/lr1e-4'; TE = 'outputs/lora_step2_eval/lr1e-4/step660'
    I = dict(n_records=428, n_mix_samples=200, mix_list='configs/mix_v1_devel100_v2.json',
             single=dict(C1=dict(P2=g('outputs/lora_base_audio/p2_devel_clean/*/P2/raw.jsonl'), W=g('outputs/lora_base_audio/whisper_t2_devel_clean/*/W/raw.jsonl'),
                                 T=g('outputs/lora_base_audio/whisper_t2_devel_clean/*/T/raw.jsonl'),
                                 AUDIO=[f'{AE}/22025280/step440/devel_clean_selection/raw.jsonl', f'{AE}/22027861/step440/devel_clean_report/raw.jsonl'],
                                 T2LORA=[f'{TE}/devel_clean_selection_22052673/T2/raw.jsonl', f'{TE}/devel_clean_report_22053122/T2/raw.jsonl']),
                         C1R=dict(P2=g('outputs/lora_base_audio/p2_devel_rev/*/P2/raw.jsonl'), W=g('outputs/lora_base_audio/whisper_t2_devel_rev/*/W/raw.jsonl'),
                                  T=g('outputs/lora_base_audio/whisper_t2_devel_rev/*/T/raw.jsonl'),
                                  AUDIO=[f'{AE}/22027862/step440/devel_rev_selection/raw.jsonl', f'{AE}/22027863/step440/devel_rev_report/raw.jsonl'],
                                  T2LORA=[f'{TE}/devel_rev_all_22053123/T2/raw.jsonl']),
                         C0=dict(T2=['outputs/qwen_omni_c0_devel428/22023517/C0/raw.jsonl'], T2LORA=['outputs/qwen_omni_c0_devel428/22023517/C0/raw.jsonl'])),
             c3=dict(P2=['outputs/qwen_omni_c3_p2/22014089/C3/raw.jsonl'], AUDIO=g('outputs/lora_audio_mix/lr1e-4/*/step440/c3/raw.jsonl')),
             eval_json=dict(c5_base=str(o / 'c5_base/eval.json'), c5_audio=str(o / 'c5_lora/eval.json'), real_base=str(o / 'real_base/eval_c6_real.json'),
                            real_audio=str(o / 'real_lora/eval_c6_real.json'), real_base_te92=str(o / 'real_base/eval_c6_real.json'),
                            real_audio_te92=str(o / 'real_lora/eval_c6_real.json'), checks_audio_vs_t2=str(o / 'checks_audio_vs_t2_ci9833/checks.json'),
                            checks_audio_vs_t2_te92=str(o / 'checks_audio_vs_t2_ci9833/checks.json'), checks_p2=str(o / 'checks_p2/checks.json'),
                            checks_t2=str(o / 'checks_t2/checks.json')))
    (out / 'smoke_inputs.json').write_text(json.dumps(I, indent=1) + '\n')
    s = run('qwen_omni_lora', 'scripts/lora/score_slurp_test_v1.py', '--inputs', o / 'smoke_inputs.json', '--out', o / 'smoke_report')
    rp = json.load(open(out / 'smoke_report/report.json'))['primary']
    cmp = json.load(open(PROJ / 'outputs/lora_step2_eval/audio_vs_t2_comparison_v2/comparison.json'))
    good = all(rp[c]['delta'] == cmp[f'{c}_all']['paired_audio_minus_t2']['delta'] and rp[c]['ci95'] == cmp[f'{c}_all']['paired_audio_minus_t2']['ci95'] for c in ('C1', 'C1R'))
    good &= rp['C5']['delta'] == 0.09 and rp['C5']['ci95'] == [0.05, 0.13]
    ok('score_slurp_test_v1_smoke_devel_primary_reproduced', good, {c: [rp[c]['delta'], rp[c]['ci95']] for c in rp})


def dryrun(out):
    o = out.relative_to(PROJ)
    s = run('qwen_omni_lora', 'scripts/qwen/qwen_omni_snapshot.py', '--model', 'spatialnet', '--final', '--sep-dir', SEP,
            '--selection', 'outputs/frontend_final_eval/ep100_20261107/spatialnet/selection.jsonl', '--out', o / 'snap', '--dry-run', '2', flags=('-B',))
    prov = json.loads(next(x for x in s.splitlines() if x.startswith('PROVENANCE '))[11:])
    rec = json.load(open(PROJ / 'outputs/qwen_omni_final/22029639_selected/config.json'))['audio_sha256']
    ok('snapshot_audio_sha256_selected', prov['audio_sha256'] == rec, dict(n=len(prov['audio_sha256'])))
    parts = {}
    for k in (0, 1):
        s = run('qwen_omni_lora', 'scripts/qwen/qwen_omni_snapshot.py', '--model', 'spatialnet', '--final', '--sep-dir', SEP,
                '--selection', 'outputs/frontend_final_eval/ep100_20261107/spatialnet/other_selection.jsonl', '--out', o / f'snap_other_p{k}',
                '--dry-run', '1', '--part', str(k), '--nparts', '2', flags=('-B',))
        parts.update(json.loads(next(x for x in s.splitlines() if x.startswith('PROVENANCE '))[11:])['audio_sha256'])
    rec = json.load(open(PROJ / 'outputs/qwen_omni_final/22029640_other/config.json'))['audio_sha256']
    ok('snapshot_audio_sha256_other_two_parts', parts == rec, dict(n=len(parts)))
    run('qwen_omni_lora', 'scripts/qwen/qwen_omni_c3_p2.py', '--audio-dir', AUDIO, '--mix-dir', MIXD, '--out', o / 'c3', '--dry-run', '3', flags=('-B',))
    base = {json.loads(x)['file']: json.loads(x) for x in open(PROJ / 'outputs/qwen_omni_c3_p2/22014089/C3/raw.jsonl')}
    dr = sorted((out / 'c3').glob('dry_C3_*.txt'))
    ok('c3_prompt_text', len(dr) == 3 and all(p.read_text() == base[p.stem[7:]]['prompt_text'] for p in dr), [p.name for p in dr])
    s = run('qwen_omni_lora', 'scripts/lora/eval_audio_lora_mix.py', '--run-dir', SH / 'checkpoints/lora_v2/step1/lr1e-4', '--step', '440', '--cond', 'c3',
            '--out', o / 'c3lora', '--dry-run', flags=('-B',))
    ok('c3_lora_prompt_equals_baseline', 'PROMPT_TEXT_EQUALS_BASELINE 5 / 5' in s)
    s = run('qwen_omni_lora', 'scripts/lora/eval_audio_lora_ckpt.py', '--run-dir', SH / 'checkpoints/lora_v2/step1/lr1e-4', '--step', '440', '--items', 'devel_clean',
            '--half', 'selection', '--out', o / 'ackpt', '--dry-run', flags=('-B',))
    ok('audio_lora_ckpt_items', 'ITEMS 214 devel_clean selection' in s and 'DRY_RUN_PASS' in s)
    s = run('qwen_omni_lora', 'scripts/lora/eval_step2_lora.py', '--run-dir', SH / 'checkpoints/lora_v2/step2/lr1e-4', '--step', '660', '--items', 'devel_clean',
            '--half', 'selection', '--asr', *sorted(PROJ.glob('outputs/lora_base_audio/whisper_t2_devel_clean/*/W/raw.jsonl')), '--out', o / 't2', '--dry-run', flags=('-B',))
    ok('t2lora_items', 'ITEMS 214 of 214' in s and 'DRY_RUN_PASS' in s)
    s = run('qwen_omni_lora', 'scripts/lora/baseline_audio.py', '--mode', 'p2', '--items', 'devel_rev', '--part', '0', '--nparts', '4', '--out', o / 'base', '--dry-run', '1', flags=('-B',))
    ok('baseline_audio_items', 'AUDIO_LOADED 107 of 428' in s and 'DRY_RUN_PASS' in s)
    s = run('qwen_omni_lora', 'scripts/qwen/qwen_omni_c0_devel428.py', '--out', o / 'c0', '--dry-run', '1', flags=('-B',))
    ok('c0_items', 'INPUTS_BUILT 428' in s and 'DRY_RUN_PASS' in s)
    # constants resolved by the split switch equal the old literals (scripts whose devel rerun needs a GPU)
    s = run('qwen_omni_lora', '-c', 'import sys; sys.path[:0]=["scripts/qwen","scripts/sms_wsj"]; import split_cfg as S, qwen_omni_c3_p2 as C, eval_c6_real as R; '
            'print(str(S.MIX), str(C.LIST), R.MANIFEST, str(S.GEOM), str(S.DIRECT_REFS))', flags=('-B',))
    exp = '<SHARED_ROOT>/data/mix_v1/devel_seed2026 ' + str(PROJ / 'configs/mix_v1_devel100_v2.json') + \
          ' <SHARED_ROOT>/data/mix_v1/devel_seed2026/manifest.jsonl ' + str(PROJ / 'results/sms_wsj_geometry/cv_dev93_per_mixture.csv') + \
          ' ' + str(PROJ / 'outputs/frontend_mix_v1_eval_v2')
    ok('split_cfg_devel_constants', s.strip() == exp, s.strip())


def main(step, out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(['git', '-C', str(PROJ), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(PROJ), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
    try:
        dict(construct=construct, select=select, direct=direct, score=score, dryrun=dryrun)[step](out)
    finally:
        rep = dict(step=step, commit=commit, dirty=dirty.splitlines(), job=os.environ.get('SLURM_JOB_ID'), checks=RES, passed=bool(RES) and all(v['passed'] for v in RES.values()))
        (out / 'check.json').write_text(json.dumps(rep, indent=1) + '\n')
    print('S0_STEP', step, 'PASS' if rep['passed'] else 'FAIL', flush=True)
    sys.exit(0 if rep['passed'] else 1)


if __name__ == '__main__':
    main(*sys.argv[1:3])
