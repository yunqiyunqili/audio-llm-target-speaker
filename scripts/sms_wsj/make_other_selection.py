"""Write an 'other stream' selection file (docs/c6_delta_spatialnet_v1.md): stream -> 1 - stream, all other fields unchanged.
Usage: python3 scripts/sms_wsj/make_other_selection.py IN_SELECTION OUT_SELECTION
"""
import json
import sys
from pathlib import Path


def main(src, dst):
    rows = [json.loads(l) for l in open(src)]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    assert not Path(dst).exists(), dst
    with open(dst, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(dict(r, stream=1 - r['stream'], source_selection=str(src))) + '\n')
    print('WROTE', len(rows), dst)


if __name__ == '__main__':
    main(*sys.argv[1:3])
