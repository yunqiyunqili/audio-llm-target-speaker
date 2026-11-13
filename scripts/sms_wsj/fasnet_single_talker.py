"""F1 step 1 and R1-same audio (docs/fasnet_single_talker_v1.md, frozen). For each mix_v1 sample: the TARGET's full
6-mic reverberant image only (per-sample RIR position, manifest gain / rir_start, list offset; no interferer, no noise),
recomputed in memory with mix_slurp_v1 functions.
  R1-same : mic 0 of that image (8 kHz) -> OUT/r1same/<key>.wav
  F1      : FaSNet-TAC (read-only checkpoint, inference lines identical to frontend_mix_v1.py) on the 6-mic image;
            stream with higher RMS (no reference), polarity aligned to mic 0 by zero-lag correlation -> OUT/streams/<key>.wav
Also records the chosen stream's SI-SDR vs the target direct reference (description only) and the two-stream RMS ratio.
Writes only to OUT (outputs/...). Usage (tfgridnet env): python -B scripts/sms_wsj/fasnet_single_talker.py MIX_DIR SLURP_DIR V2_DIR OUT [--n N]
"""
import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import fftconvolve, resample_poly  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
MC = Path('<SEPARATION_REPO>/multichannel')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend_mix_v1 import RUNS, sha  # noqa: E402
from mix_slurp_v1 import extract_piece  # noqa: E402
from eval_frontend_mix_v1 import sisdr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mix_dir', type=Path); ap.add_argument('slurp_dir', type=Path); ap.add_argument('v2dir', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    run = RUNS['fasnet_tac']; sys.path.insert(0, str(MC / 'fasnet_tac'))
    import torch
    import train as T
    rec = json.loads((run / 'test_summary.json').read_text()); cfg = json.loads((run / 'run_config.json').read_text())
    pkg = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    assert sha(run / 'best.pt') == rec['checkpoint_sha256'] and pkg['config'] == cfg
    torch.set_num_threads(4); T.seed_all()
    model = T.Model(6); model.load_state_dict(pkg['model']); model.eval()
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = [json.loads(l) for l in open(a.mix_dir / 'manifest.jsonl')]
    man = sorted(man, key=lambda m: (m['group'], m['swap']))[:a.n or None]
    a.out.mkdir(parents=True, exist_ok=False); (a.out / 'streams').mkdir(); (a.out / 'r1same').mkdir()
    cache, rows = {}, []
    with torch.no_grad():
        for m in man:
            g = L[m['group']]; key = f"g{m['group']:03d}_s{m['swap']}"; t = m['target']; T8 = m['T_8k']; info = m['sources'][t]
            f = g[t]['file']
            if f not in cache:
                cache[f] = resample_poly(sf.read(a.slurp_dir / f, dtype='float64')[0], 1, 2)
            h = sf.read(Path(g['rir']['rir_dir']) / f"h_{info['pos']}.wav", dtype='float64', always_2d=True)[0].T
            img = extract_piece(fftconvolve(cache[f][None], h, axes=-1), g[t]['offset_8k'] - info['rir_start'], T8) * info['gain']   # (6, T)
            sf.write(a.out / 'r1same' / f'{key}.wav', img[0].astype(np.float32), 8000, subtype='FLOAT')
            x = img.T.astype(np.float32)
            mix = torch.from_numpy(x)[None]
            xin, _, scale = T.normalize(mix, mix[..., :2].transpose(1, 2))
            y = (model(xin) * scale)[0].numpy().astype(np.float64)        # identical to frontend_mix_v1.py
            rms = np.sqrt((y ** 2).mean(1)); k = int(np.argmax(rms))
            sign = -1.0 if float((y[k] * x[:, 0]).sum()) < 0 else 1.0
            s = sign * y[k]
            sf.write(a.out / 'streams' / f'{key}.wav', s.astype(np.float32), 8000, subtype='FLOAT')
            d = np.load(a.v2dir / 'direct_refs' / f'{key}.npz')[t].astype(np.float64)
            rows.append(dict(key=key, target=t, pos=info['pos'], stream=k, sign=sign, rms_ratio_low_high=float(rms.min() / max(rms.max(), 1e-20)),
                             sisdr_direct=sisdr(d, s), peak_r1same=float(np.abs(img[0]).max())))
            print('DONE', key, k, sign, round(rows[-1]['sisdr_direct'], 2), flush=True)
    with open(a.out / 'f1_streams.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print('SUMMARY', json.dumps(dict(n=len(rows), stream0=sum(r['stream'] == 0 for r in rows), flipped=sum(r['sign'] < 0 for r in rows),
                                     median_sisdr_direct=float(np.median([r['sisdr_direct'] for r in rows])),
                                     median_rms_ratio=float(np.median([r['rms_ratio_low_high'] for r in rows])))), flush=True)


if __name__ == '__main__':
    main()
