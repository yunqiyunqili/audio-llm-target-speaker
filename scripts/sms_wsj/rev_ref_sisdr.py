"""SI-SDR of BLSTM+MVDR / oracle-MVDR / cACGMM-MVDR outputs against three mic-0 references
on test_eval92: direct path (as recorded), early (direct + 50 ms) image, full reverberant
image (early+tail). PIT over 2 speakers, zero-mean SI-SDR from tfgridnet/objectives.py.

* BLSTM outputs were never saved -> re-run the frozen best.pt on GPU (scores only, no audio).
* oracle / cACGMM outputs are read from their saved WAVs (read-only).
Nothing is written under ../speech_separation. NOT SUBMITTED; see rev_ref_sisdr.sbatch.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json
from pathlib import Path

MC = Path('<SEPARATION_REPO>/multichannel')
SAVED = {'oracle_mvdr': MC/'oracle_mvdr/exp/21977751/test/audio',
         'cacgmm_mvdr': MC/'spatial_baseline/exp/full_21976118/test/audio'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=None, help='first N utterances (debug)')
    args = ap.parse_args()
    sys.path.insert(0, str(MC/'blstm_mvdr'))
    import numpy as np, soundfile as sf, torch
    import train as T                      # original BLSTM code (dataset, model, seed_all)
    from tfgridnet.objectives import normalize, sisdr

    run = MC/'blstm_mvdr/exp/sms_blstm_mvdr_v1'
    cfg = json.loads((run/'run_config.json').read_text())
    ds = T.SMSWSJ(Path(cfg['data_root']), 'test_eval92', tuple(range(6)), segment=None)
    T.seed_all(); torch.set_num_threads(4)
    model = T.Model(6).cuda().eval()
    model.load_state_dict(torch.load(run/'best.pt', map_location='cpu', weights_only=False)['model'])

    def pit(refs, est):  # refs/est [2,T] tensors -> {name: score}
        out = {}
        for name, ref in refs.items():
            v = torch.stack([sisdr(ref, est), sisdr(ref, est.flip(0))]).mean(-1)
            out[name] = float(v.max())
        return out

    args.out.mkdir(parents=True, exist_ok=False)
    n = len(ds) if args.limit is None else args.limit
    rows = []
    with torch.no_grad(), (args.out/'utterances.jsonl').open('w') as f:
        for i in range(n):
            item = ds[i]; rec = ds.records[i]; key = item['id']; L = item['length']
            base = Path(rec['mixture']).parents[2]
            early = torch.stack([torch.from_numpy(sf.read(base/f'early/test_eval92/{key}_{k}.wav', dtype='float32')[0][:, 0])
                                 for k in range(2)])
            refs = dict(direct=item['target'][:, :L].double(), early=early[:, :L].double(),
                        reverberant=item['components'][:2, 0, :L].double())
            mix = item['mixture'][None].cuda()
            x, _, scale = normalize(mix, item['target'][None].cuda())
            est = {'blstm_mvdr': (model(x)*scale)[0, :, :L].cpu().double()}
            for name, d in SAVED.items():
                est[name] = torch.stack([torch.from_numpy(sf.read(d/f'{key}_{k}.wav', dtype='float32')[0])
                                         for k in range(2)])[:, :L].double()
            row = dict(id=key, samples=L, mixture={r: float(sisdr(v, mix[0, :L, 0].cpu().double().expand_as(v)).mean())
                                                    for r, v in refs.items()})
            row.update({name: pit(refs, e) for name, e in est.items()})
            rows.append(row); f.write(json.dumps(row)+'\n')
            if (i+1) % 100 == 0: print('DONE', i+1, flush=True)
    systems = ['mixture', 'blstm_mvdr', *SAVED]
    summary = {s: {r: float(np.mean([row[s][r] for row in rows])) for r in ('direct', 'early', 'reverberant')} for s in systems}
    summary['utterances'] = len(rows)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('SUMMARY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
