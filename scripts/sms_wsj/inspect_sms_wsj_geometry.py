"""Read sms_wsj.json (read-only), print geometry for a few examples, check RIR files.

Azimuth convention: horizontal angle atan2(dy, dx) of (source - array centre) in the
SMS-WSJ room coordinate frame (x/y axes of the room). Also printed relative to the
direction centre->mic0, which removes the per-room random array rotation.
Light CPU job (~2 s, ~2 GB RAM). Usage:
  python -I scripts/sms_wsj/inspect_sms_wsj_geometry.py [split/example_id ...]
"""
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

DB = '<SEPARATION_DATA_ROOT>/data/sms_wsj/sms_wsj.json'


def azimuth(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))


def wrap(deg):
    return (deg + 180) % 360 - 180


def main(picks):
    ds = json.load(open(DB))['datasets']
    if not picks:  # deterministic default: two test examples, one dev example
        picks = [f'test_eval92/{k}' for k in sorted(ds['test_eval92'])[:2]] + \
                [f'cv_dev93/{sorted(ds["cv_dev93"])[0]}']
    for pick in picks:
        split, key = pick.split('/', 1)
        e = ds[split][key]
        src = np.asarray(e['source_position'], float)   # [3, n_speakers]
        mic = np.asarray(e['sensor_position'], float)   # [3, 6]
        centre = mic.mean(1)
        ref = azimuth(mic[:, 0] - centre)
        print(f'== {pick}  room={np.ravel(e["room_dimensions"]).tolist()}  T60={e["sound_decay_time"]:.3f}s'
              f'  array_radius={np.linalg.norm(mic - centre[:, None], axis=0).mean():.4f}m')
        azs = []
        for s in range(src.shape[1]):
            d = src[:, s] - centre
            az = azimuth(d); azs.append(az)
            el = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
            print(f'  spk{s} id={e["speaker_id"][s]} az_room={az:.2f} az_rel_mic0={wrap(az - ref):.2f}'
                  f' elev={el:.2f} dist={np.linalg.norm(d):.3f}m')
        if len(azs) == 2:
            print(f'  azimuth separation={abs(wrap(azs[1] - azs[0])):.2f} deg')
        for p in e['audio_path']['rir']:
            info = sf.info(p) if os.path.isfile(p) else None
            print(f'  RIR {p} exists={info is not None}' +
                  (f' sr={info.samplerate} ch={info.channels} frames={info.frames}' if info else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
