"""External check of the v3 polarity rule on SMS-WSJ cv_dev93 (docs/frontend_mix_v1_eval_v3.md, frozen).
50 official mixtures (random.Random(2026).sample(sorted ids, 50)); front-end inference in memory (same lines as
frontend_mix_v1.py; no audio written); direct references = official sms_wsj_direct_v1 files; two target samples per
mixture (target = source 0, target = source 1). Reports stream-vs-own-direct zero-lag correlation by stream index,
flip rate, and direction-matching accuracy before / after the polarity rule (estimator and tie rule unchanged).
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/polarity_check_cv_dev93.py --model M --out DIR
"""
import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
MC = Path('<SEPARATION_REPO>/multichannel')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend_mix_v1 import RUNS, sha  # noqa: E402
from eval_frontend_mix_v1 import DB, doa, pit, wrap  # noqa: E402
from eval_frontend_mix_v1_v3 import choose, polarity_fix  # noqa: E402

DIRECT = MC / 'data/sms_wsj_direct_v1/cv_dev93'
GEOM = PROJ / 'results/sms_wsj_geometry/cv_dev93_per_mixture.csv'


def stratum(sep):
    return '[0,15)' if sep < 15 else '[15,30)' if sep < 30 else '[30,60)' if sep < 60 else '[60,180]'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=sorted(RUNS), required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); run = RUNS[a.model]
    sys.path.insert(0, str(MC / a.model))
    import torch
    import train as T
    rec = json.loads((run / 'test_summary.json').read_text()); cfg = json.loads((run / 'run_config.json').read_text())
    pkg = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    assert sha(run / 'best.pt') == rec['checkpoint_sha256'] and pkg['config'] == cfg
    torch.set_num_threads(4); T.seed_all()
    model = T.Model(6); model.load_state_dict(pkg['model']); model.eval()
    db = json.load(open(DB))['datasets']['cv_dev93']
    geo = {r['id']: r for r in csv.DictReader(open(GEOM))}
    ids = random.Random(2026).sample(sorted(db), 50)
    rows = []
    with torch.no_grad():
        for eid in ids:
            e = db[eid]
            x, sr = sf.read(e['audio_path']['observation'], dtype='float32', always_2d=True); assert sr == 8000
            mix = torch.from_numpy(x)[None]
            xin, _, scale = T.normalize(mix, mix[..., :2].transpose(1, 2))
            y = (model(xin) * scale)[0].numpy().astype(np.float64)          # identical to frontend_mix_v1.py
            obs = x.T.astype(np.float64)
            D = [sf.read(DIRECT / f'{eid}_{k}.wav', dtype='float64', always_2d=True)[0][:, 0] for k in range(2)]
            mic = np.asarray(e['sensor_position'], float); c = mic.mean(1); ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
            th = [float(geo[eid]['az_rel_mic0_0']), float(geo[eid]['az_rel_mic0_1'])]
            p, _, _ = pit(D, y)                                               # p[k] = stream holding source k
            corr_own = {p[k]: float(np.dot(y[p[k]], D[k]) / (np.linalg.norm(y[p[k]]) * np.linalg.norm(D[k]) + 1e-20)) for k in range(2)}
            yf, flipped = polarity_fix(y, obs[0])
            az0 = [wrap(doa(y[k], obs, mic[:2].T) - ra) for k in range(2)]
            az1 = [wrap(doa(yf[k], obs, mic[:2].T) - ra) for k in range(2)]
            for tgt in range(2):
                ts = p[tgt]
                rows.append(dict(id=eid, target_source=tgt, stratum=stratum(float(geo[eid]['separation'])),
                                 corr_own_stream=[corr_own[0], corr_own[1]], flipped=[bool(f) for f in flipped],
                                 ref_stream=ts, correct_before=choose(az0, th[tgt]) == ts, correct_after=choose(az1, th[tgt]) == ts,
                                 err_target_before=abs(wrap(az0[ts] - th[tgt])), err_target_after=abs(wrap(az1[ts] - th[tgt]))))
            print('DONE', eid, flush=True)
    G = [r for r in rows if r['stratum'] != '[0,15)']
    q = lambda sel, f: float(np.mean([f(r) for r in sel])) if sel else None  # noqa: E731
    per_mix = rows[::2]
    rep = dict(model=a.model, n_mixtures=len(ids), n_target_samples=len(rows), n_excl_0_15=len(G),
               corr_own_negative_stream0=q(per_mix, lambda r: r['corr_own_stream'][0] < 0), corr_own_negative_stream1=q(per_mix, lambda r: r['corr_own_stream'][1] < 0),
               median_corr_own_stream0=float(np.median([r['corr_own_stream'][0] for r in per_mix])),
               median_corr_own_stream1=float(np.median([r['corr_own_stream'][1] for r in per_mix])),
               flip_rate_stream0=q(per_mix, lambda r: r['flipped'][0]), flip_rate_stream1=q(per_mix, lambda r: r['flipped'][1]),
               acc_before_all=q(rows, lambda r: r['correct_before']), acc_after_all=q(rows, lambda r: r['correct_after']),
               acc_before_excl_0_15=q(G, lambda r: r['correct_before']), acc_after_excl_0_15=q(G, lambda r: r['correct_after']),
               median_err_target_before=float(np.median([r['err_target_before'] for r in rows])),
               median_err_target_after=float(np.median([r['err_target_after'] for r in rows])),
               strata_counts={s: sum(r['stratum'] == s for r in per_mix) for s in ('[0,15)', '[15,30)', '[30,60)', '[60,180]')})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f'polarity_cv_dev93_{a.model}.json').write_text(json.dumps(dict(summary=rep, rows=rows), indent=1) + '\n')
    print('SUMMARY', json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
