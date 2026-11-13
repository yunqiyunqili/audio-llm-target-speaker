"""C5 overlap-damage diagnostic (docs/c5_overlap_damage_v1.md, frozen; read-only; descriptive, not causal; no new audio).
A: Whisper S/D/I decomposition for W_C4direct vs W_C5 (normalize_v2; Levenshtein backtrace, ties S > D > I; total checked
   against slu_normalize.wer_counts); per-item comparison; top-10 W_C5 WER items with main error type.
B: per-frame classes on the C5 target stream (256-sample frames; activity per direct reference, >= peak - 40 dB):
   target-only vs overlap; LS split y = e_t + e_i + e_r; class-level target/interference, target/residual, segmental SDR.
Usage (tfgridnet env, CPU): python -B scripts/sms_wsj/c5_overlap_damage.py SEP_ROOT V2_DIR RUN_22016221 C5_EVAL_PER_SAMPLE_SRC OUT_DIR
  C5_EVAL_PER_SAMPLE_SRC = outputs/qwen_omni_c5/22015337/C5_fasnet_tac/raw.jsonl (hit computed with eval_c4_v1.score)
"""
import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/slurp'))
from rescore_slu_v2 import normalize_v2  # noqa: E402
from slu_normalize import wer_counts  # noqa: E402
from eval_c4_v1 import score  # noqa: E402

FRAME, ACTIVE_DB = 256, 40.0


def sdi(ref, hyp):
    r, h = ref.split(), hyp.split(); n, m = len(r), len(h)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (r[i - 1] != h[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, S, De, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            De += 1; i -= 1
        else:
            I += 1; j -= 1
    assert S + De + I == D[n, m]
    return S, De, I, n


def active(x):
    n = len(x) // FRAME
    p = (x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1)
    return p >= p.max() * 10 ** (-ACTIVE_DB / 10)


def db(a, b):
    return float(10 * np.log10(max(a, 1e-30) / max(b, 1e-30)))


def q(x):
    x = [v for v in x if v is not None]
    return dict(n=len(x), p25=float(np.percentile(x, 25)), median=float(np.median(x)), p75=float(np.percentile(x, 75))) if x else dict(n=0)


def main(sep_root, v2dir, run, c5raw, out):
    sep_root, v2dir, run, out = Path(sep_root), Path(v2dir), Path(run), Path(out); out.mkdir(parents=True, exist_ok=False)
    L = {g['group']: g for g in json.load(open(PROJ / 'configs/mix_v1_devel100_v2.json'))['groups']}
    man = {f"g{r['group']:03d}_s{r['swap']}": r for r in map(json.loads, open('<SHARED_ROOT>/data/mix_v1/devel_seed2026/manifest.jsonl'))}
    hit = {r['key']: r['t_any'] for r in score({json.loads(l)['file']: json.loads(l) for l in open(c5raw)}, L, man)}
    W = {s: {json.loads(l)['file']: json.loads(l) for l in open(run / f'W_{s}' / 'raw.jsonl')} for s in ('C4direct', 'C5')}
    sel = {r['key']: r for r in map(json.loads, open(PROJ / 'outputs/c5_v1/selection.jsonl'))}
    # ---- A
    tot = {s: np.zeros(4, int) for s in W}; per = []
    for k, m in man.items():
        g = L[m['group']]; t = m['target']; i = 'B' if t == 'A' else 'A'; ref = normalize_v2(g[t]['sentence'])
        row = dict(key=k, hit=bool(hit[k]))
        for s in W:
            h = normalize_v2(W[s][k]['raw_output']); S_, D_, I_, n = sdi(ref, h)
            assert S_ + D_ + I_ == wer_counts(ref, h)[0]
            tot[s] += [S_, D_, I_, n]; row[s] = dict(S=S_, D=D_, I=I_, n=n, wer=(S_ + D_ + I_) / n, hyp=W[s][k]['raw_output'],
                                                     wer_interferer=wer_counts(normalize_v2(g[i]['sentence']), h)[0] / len(normalize_v2(g[i]['sentence']).split()))
        row['ref'] = g[t]['sentence']; row['interferer'] = g[i]['sentence']; per.append(row)
    A = {s: dict(S=int(v[0]), D=int(v[1]), I=int(v[2]), ref_words=int(v[3]), S_rate=v[0] / v[3], D_rate=v[1] / v[3], I_rate=v[2] / v[3],
                 wer=(v[0] + v[1] + v[2]) / v[3]) for s, v in tot.items()}
    dw = [r['C5']['wer'] - r['C4direct']['wer'] for r in per]
    A['per_item'] = dict(C5_worse=sum(x > 1e-12 for x in dw), equal=sum(abs(x) <= 1e-12 for x in dw), C5_better=sum(x < -1e-12 for x in dw),
                         delta_S=int(tot['C5'][0] - tot['C4direct'][0]), delta_D=int(tot['C5'][1] - tot['C4direct'][1]), delta_I=int(tot['C5'][2] - tot['C4direct'][2]))
    for s in ('C4direct', 'C5'):
        A[f'{s}_by_hit'] = {lab: dict(wer=float(np.mean([r[s]['wer'] for r in per if r['hit'] == h_])), n=sum(r['hit'] == h_ for r in per))
                            for lab, h_ in (('hit', True), ('miss', False))}
    top = sorted(per, key=lambda r: -r['C5']['wer'])[:10]

    def main_type(r):
        c = r['C5']
        if c['wer_interferer'] < c['wer']:
            t = 'toward interferer'
        else:
            t = max((('substitution', c['S']), ('deletion', c['D']), ('insertion', c['I'])), key=lambda x: x[1])[0]
        return t + (' (massive insertion)' if c['I'] > c['n'] else '')
    A['top10'] = [dict(key=r['key'], ref=r['ref'], w_c5=r['C5']['hyp'], S=r['C5']['S'], D=r['C5']['D'], I=r['C5']['I'], wer=r['C5']['wer'],
                       wer_to_interferer=r['C5']['wer_interferer'], main_error=main_type(r), c5_hit=r['hit']) for r in top]
    # ---- B
    B = []
    for k, m in man.items():
        t = m['target']; i = 'B' if t == 'A' else 'A'; s = sel[k]; st = s['stream']
        y = sf.read(sep_root / 'fasnet_tac' / k / f'out{st}.wav', dtype='float64')[0] * (-1.0 if s['flipped'][st] else 1.0)
        dz = np.load(v2dir / 'direct_refs' / f'{k}.npz'); dT, dI = dz[t].astype(np.float64), dz[i].astype(np.float64)
        yc, Tc, Ic = y - y.mean(), dT - dT.mean(), dI - dI.mean()
        c, *_ = np.linalg.lstsq(np.stack([Tc, Ic], 1), yc, rcond=None)
        et, ei = c[0] * Tc, c[1] * Ic; er = yc - et - ei
        aT, aI = active(dT), active(dI); nf = min(len(aT), len(aI))
        cls = dict(target_only=aT[:nf] & ~aI[:nf], overlap=aT[:nf] & aI[:nf])
        fr = lambda x: (x[:nf * FRAME].reshape(nf, FRAME) ** 2).sum(1)  # noqa: E731
        Et, Ei, Er, Ey = fr(et), fr(ei), fr(er), fr(yc - et)
        row = dict(key=k, hit=bool(hit[k]), overlap_frac=float(cls['overlap'].sum() / max((aT[:nf] | aI[:nf]).sum(), 1)))
        for name, mask in cls.items():
            if mask.sum() == 0:
                row[name] = None; continue
            row[name] = dict(frames=int(mask.sum()), t_over_i=db(Et[mask].sum(), Ei[mask].sum()), t_over_r=db(Et[mask].sum(), Er[mask].sum()),
                             seg_sdr=db(Et[mask].sum(), Ey[mask].sum()))
        B.append(row)

    def bsum(rows):
        o = dict(n=len(rows), overlap_frac=q([r['overlap_frac'] for r in rows]))
        for name in ('target_only', 'overlap'):
            o[name] = {mt: q([r[name][mt] if r[name] else None for r in rows]) for mt in ('t_over_i', 't_over_r', 'seg_sdr')}
            o[name]['samples_with_frames'] = sum(r[name] is not None for r in rows)
        return o
    Bs = dict(all=bsum(B), c5_hit=bsum([r for r in B if r['hit']]), c5_miss=bsum([r for r in B if not r['hit']]))
    rep = dict(spec='docs/c5_overlap_damage_v1.md', note='descriptive only; not causal', A=A, B=Bs)
    (out / 'overlap_damage.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    with open(out / 'per_sample.jsonl', 'w') as fh:
        for a_, b_ in zip(per, B):
            fh.write(json.dumps(dict(**a_, frames=b_), ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in A.items() if k != 'top10'}, indent=1)); print(json.dumps(Bs, indent=1))
    for x in A['top10']:
        print('TOP', json.dumps(x, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:6])
