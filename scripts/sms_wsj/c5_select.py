"""C5 v1 stream selection (docs/c5_v1.md, frozen). Uses ONLY: the two FaSNet-TAC output streams, the 6-mic observation,
the array geometry (sensor positions) and the query azimuth theta1 (ground-truth target azimuth = C5 definition; ORACLE).
Does NOT read direct / early / full references, speaker identity or A/B ownership. Rule = frontend-mix-v1-eval-v3:
polarity_fix (zero-lag correlation with mic 0) -> doa -> choose (closest to theta1, tie -> stream 0).
Writes OUT_DIR/selection.jsonl: key, stream, doa (both streams), flipped (both streams).
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/c5_select.py MIX_DIR SEP_ROOT OUT_DIR
"""
import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_frontend_mix_v1 import DB, doa, wrap  # noqa: E402
from eval_frontend_mix_v1_v3 import choose, polarity_fix  # noqa: E402


def main(mix_dir, sep_root, out):
    mix_dir, sep_root, out = Path(mix_dir), Path(sep_root), Path(out); out.mkdir(parents=True, exist_ok=False)
    groups = json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']
    geo = json.load(open(DB))['datasets']['cv_dev93']
    n = 0
    with open(out / 'selection.jsonl', 'w') as fh:
        for g in sorted(groups, key=lambda g: g['group']):
            theta_q = g['rir']['theta1']                                   # query azimuth (oracle, C5 definition)
            mic = np.asarray(geo[g['rir']['example_id']]['sensor_position'], float); c = mic.mean(1)
            ra = math.degrees(math.atan2(mic[1, 0] - c[1], mic[0, 0] - c[0]))
            for s in (1, 2):
                key = f"g{g['group']:03d}_s{s}"
                obs = sf.read(mix_dir / key / 'obs_6ch_8k.wav', dtype='float64', always_2d=True)[0].T
                y = np.stack([sf.read(sep_root / 'fasnet_tac' / key / f'out{k}.wav', dtype='float64')[0] for k in range(2)])
                yf, flipped = polarity_fix(y, obs[0])
                az = [wrap(doa(yf[k], obs, mic[:2].T) - ra) for k in range(2)]
                fh.write(json.dumps(dict(key=key, stream=choose(az, theta_q), theta_query=theta_q, doa=az, flipped=[bool(f) for f in flipped])) + '\n')
                n += 1
    print('SELECTED', n, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
