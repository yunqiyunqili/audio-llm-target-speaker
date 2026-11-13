"""Generate FSC-only PHAT DOA/selected-stream metadata (CPU, frozen C5 rule).

Selection uses only the 6-channel observation, separated streams, array geometry,
and the direction query. Source labels and early references are read only for
post-hoc diagnostics after selection.
"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/sms_wsj'))
from eval_frontend_mix_v1 import DB, doa, wrap  # noqa: E402
from eval_frontend_mix_v1_v3 import choose, polarity_fix  # noqa: E402

O = ROOT / 'outputs/fsc_pilot'
MIX = O / 'spatial_audio_real'
SEP = O / 'spatialnet_ep100'
OUT = O / 'route_manifest'
TAU = 16.0
DELTAS = (-20.0, -10.0, 0.0, 10.0, 20.0)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def normcorr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

def main():
    OUT.mkdir(parents=False, exist_ok=False)
    pilot = [json.loads(x) for x in open(O / 'valid64_manifest.jsonl')]
    mix = [json.loads(x) for x in open(MIX / 'manifest.jsonl')]
    assert len(pilot) == len(mix) == 64
    assert len({x['pilot_index'] for x in pilot}) == 64
    assert len({(x['group'], x['swap']) for x in mix}) == 64
    geo = json.load(open(DB))['datasets']['cv_dev93']
    code_files = [ROOT / 'scripts/sms_wsj/eval_frontend_mix_v1.py', ROOT / 'scripts/sms_wsj/eval_frontend_mix_v1_v3.py', Path(__file__)]
    code_hash = {str(p): sha(p) for p in code_files}
    rows, variants, diag = [], [], []
    for m in sorted(mix, key=lambda x: (x['group'], x['swap'])):
        key = m['dir'].split('/')[-1]
        assert key == f"g{m['group']:03d}_s{m['swap']}"
        obs_p = MIX / key / 'obs_6ch_8k.wav'
        obs, sr = sf.read(obs_p, dtype='float64', always_2d=True)
        assert sr == 8000 and obs.shape[1] == 6
        obs = obs.T
        y = []
        for k in range(2):
            yp = SEP / key / f'out{k}.wav'; yy, ys = sf.read(yp, dtype='float64')
            assert ys == 8000
            y.append(yy)
        y = np.stack(y); yf, flipped = polarity_fix(y, obs[0])
        # Geometry and reference azimuth are used only to express room DOA.
        # They do not expose speaker identity or semantic labels to selection.
        g = json.load(open(ROOT / 'configs/mix_v1_devel100_v2.json'))['groups'][m['group']]
        mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); cen = mic.mean(1)
        ref_az = math.degrees(math.atan2(mic[1, 0] - cen[1], mic[0, 0] - cen[0]))
        az = [wrap(doa(yf[k], obs, mic[:2].T) - ref_az) for k in range(2)]
        theta = float(m['theta_question']); selected = choose(az, theta)
        base = dict(sample_id=key, pair_id=f"g{m['group']:03d}", swap=int(m['swap']),
                    target_direction_deg=theta, doa_deg=az, selected_stream=selected,
                    selected_audio_path=str((SEP / key / f'out{selected}.wav').resolve()),
                    input_audio_path=str(obs_p.resolve()), input_audio_sha256=sha(obs_p),
                    output_audio_sha256=sha(SEP / key / f'out{selected}.wav'),
                    flipped=[bool(x) for x in flipped],
                    route_rule='polarity_fix_zero_lag_mic0 -> PHAT_DOA_360deg -> nearest_query_tie_stream0',
                    query_source='spatial_audio_real manifest theta_question',
                    c6_tau_deg=TAU, code_sha256=code_hash)
        rows.append(base)
        # Direction-uncertainty metadata is frozen before any semantic output.
        for delta in DELTAS:
            q = wrap(theta + delta); st = choose(az, q)
            d = [abs(wrap(a - q)) for a in az]
            variants.append(dict(sample_id=key, pair_id=f"g{m['group']:03d}", swap=int(m['swap']),
                                 query_delta_deg=delta, query_direction_deg=q, doa_deg=az,
                                 selected_stream=st, selected_audio_path=str((SEP / key / f'out{st}.wav').resolve()),
                                 output_audio_sha256=sha(SEP / key / f'out{st}.wav'),
                                 nearest_doa_error_deg=min(d), within_tau=bool(min(d) <= TAU),
                                 reject_rule='frozen C6 tau; semantic refusal scored downstream', c6_tau_deg=TAU,
                                 route_rule=base['route_rule'], code_sha256=code_hash))
        # Post-hoc only: reference matching is diagnostic, never selection input.
        target = 'A' if m['target'] == 'A' else 'B'
        refs = [sf.read(MIX / key / f'early_{x}_6ch_8k.wav', dtype='float64', always_2d=True)[0][:, 0] for x in 'AB']
        corr = [[normcorr(yf[k], refs[j]) for j in range(2)] for k in range(2)]
        target_stream = int(np.argmax([corr[k][0 if target == 'A' else 1] for k in range(2)]))
        diag.append(dict(sample_id=key, pair_id=f"g{m['group']:03d}", swap=int(m['swap']),
                         selected_stream=selected, reference_target_stream=target_stream,
                         route_correct=bool(selected == target_stream), doa_deg=az,
                         target_direction_deg=theta,
                         doa_error_selected_deg=float(abs(wrap(az[selected] - theta))),
                         doa_error_nearest_deg=float(min(abs(wrap(a - theta)) for a in az)),
                         posthoc_corr=corr, flipped=[bool(x) for x in flipped]))
    assert len(rows) == len(variants) // 5 == len(diag) == 64
    assert len({r['sample_id'] for r in rows}) == 64
    with open(OUT / 'route_manifest.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True) + '\n')
    with open(OUT / 'direction_uncertainty_manifest.jsonl', 'w') as f:
        for r in variants: f.write(json.dumps(r, sort_keys=True) + '\n')
    with open(OUT / 'diagnostics.jsonl', 'w') as f:
        for r in diag: f.write(json.dumps(r, sort_keys=True) + '\n')
    # Pair-swap and completeness checks are recorded separately.
    bypair = {}
    for r in rows: bypair.setdefault(r['pair_id'], []).append(r)
    pair_ok = all(len(v) == 2 and v[0]['target_direction_deg'] == v[1]['target_direction_deg'] for v in bypair.values())
    comp = sum(rows[2*i]['selected_stream'] != rows[2*i+1]['selected_stream'] for i in range(32))
    rep = dict(n=64, pairs=len(bypair), pair_swap_metadata_ok=pair_ok,
               selected_stream_complement_count=comp, selected_stream_complement_rate=comp/32,
               route_manifest_sha256=sha(OUT/'route_manifest.jsonl'),
               uncertainty_manifest_sha256=sha(OUT/'direction_uncertainty_manifest.jsonl'),
               diagnostics_sha256=sha(OUT/'diagnostics.jsonl'), code_sha256=code_hash,
               counts={'route':len(rows),'uncertainty':len(variants),'diagnostics':len(diag)},
               route_rule=rows[0]['route_rule'], c6_tau_deg=TAU)
    rep['diagnostic'] = dict(route_accuracy=float(np.mean([x['route_correct'] for x in diag])),
                             mean_selected_doa_error_deg=float(np.mean([x['doa_error_selected_deg'] for x in diag])),
                             median_selected_doa_error_deg=float(np.median([x['doa_error_selected_deg'] for x in diag])),
                             mean_nearest_doa_error_deg=float(np.mean([x['doa_error_nearest_deg'] for x in diag])),
                             within_tau=float(np.mean([x['doa_error_nearest_deg'] <= TAU for x in diag])))
    (OUT / 'summary.json').write_text(json.dumps(rep, indent=2) + '\n')
    print(json.dumps(rep, indent=2))

if __name__ == '__main__': main()
