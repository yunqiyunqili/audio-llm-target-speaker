"""Read-only level statistics of SMS-WSJ cv_dev93 mixtures on mic 0, with the mix_v1 active-frame definition
(scripts/sms_wsj/mix_slurp_v1.py: active_power; 256-sample frames, power >= peak frame power - 40 dB).
Per mixture: active-frame RMS of the observation (mic 0), in dBFS; also of each speaker's early image (mic 0)
for a like-for-like comparison with the mix_v1 per-source level (-30 dBFS on the mic-0 early image).
Usage: python -I scripts/sms_wsj/cv_dev93_level_stats.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_slurp_v1 import active_power  # noqa: E402

DB = '<SEPARATION_DATA_ROOT>/data/sms_wsj/sms_wsj.json'


def db(p):
    return 10 * np.log10(p)


def desc(x):
    x = np.asarray(x)
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=len(x), mean=float(x.mean()), min=float(q[0]), p5=float(q[1]), p25=float(q[2]), median=float(q[3]),
                p75=float(q[4]), p95=float(q[5]), max=float(q[6]))


def main(out):
    ds = json.load(open(DB))['datasets']['cv_dev93']
    obs, early = [], []
    for k in sorted(ds):
        ap = ds[k]['audio_path']
        x, sr = sf.read(ap['observation'], dtype='float64', always_2d=True); assert sr == 8000
        obs.append(db(active_power(x[:, 0])[0]))
        for p in ap['speech_reverberation_early']:
            e, _ = sf.read(p, dtype='float64', always_2d=True)
            early.append(db(active_power(e[:, 0])[0]))
    rep = dict(source=DB, n_mixtures=len(obs), definition='mic 0, 256-sample frames, active >= peak - 40 dB, RMS over active frames',
               observation_mic0_active_dbfs=desc(obs), early_image_per_speaker_mic0_active_dbfs=desc(early),
               mix_v1_level_dbfs=-30.0, diff_vs_observation_median_db=-30.0 - float(np.median(obs)),
               diff_vs_early_median_db=-30.0 - float(np.median(early)), rule='keep -30 dBFS if |diff vs median| <= 6 dB')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
