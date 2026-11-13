"""CPU assertion check: every FSC pilot task is preprocessed exactly like its SLURP counterpart (login node, no job).

Checks per condition: sample rate, length, level, polarity, number of chain8 applications, using the same
pilot_audio.prepare() that qwen_pilot_batch.py calls. Also checks that reverb_single.reverb() reproduces an existing
SLURP devel C1-R file, so calling it here is the C1-R chain. Writes outputs/fsc_pilot/preprocess_check.json.
"""
import collections, hashlib, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
for d in ('scripts/fsc', 'scripts/lora', 'scripts/qwen'):
    sys.path.insert(0, str(ROOT / d))
from pilot_audio import prepare  # noqa: E402
import reverb_single as RV  # noqa: E402
from qwen_omni_c4 import level, LEVEL_DBFS, FRAME16, ACTIVE_DB  # noqa: E402

O = ROOT / 'outputs/fsc_pilot'; Q = 1 / 32768


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def active_dbfs(x):
    n = len(x) // FRAME16; p = (np.asarray(x, np.float64)[:n * FRAME16].reshape(n, FRAME16) ** 2).mean(1)
    return float(10 * np.log10(p[p >= p.max() * 10 ** (-ACTIVE_DB / 10)].mean()))


def main():
    out = {}
    # 0. reverb_single.reverb() reproduces an existing SLURP devel C1-R file (16-bit) -> it is the C1-R chain.
    DREV = Path('<SHARED_ROOT>/data/lora_v1/devel_rev_cv_dev93')
    ref = [json.loads(x) for x in open(DREV / 'manifest_part0.jsonl')][:3]
    ref_err = []
    for r in ref:
        y, _ = RV.reverb(r['file'], r['rir_set'], r['rir_example'], r['rir_pos'])
        z, sr = sf.read(DREV / r['wav'], dtype='float64')
        assert sr == 16000 and len(z) == len(y); ref_err.append(float(np.abs(z - y).max())); assert ref_err[-1] <= Q
    out['slurp_c1r_reproduced'] = dict(n=len(ref), max_abs_err=max(ref_err))

    T = [json.loads(x) for x in open(O / 'pilot_tasks.jsonl')]
    assert len(T) == 256 and len({(t['condition'], t['pilot_index']) for t in T}) == 256
    M = {m['pilot_index']: m for m in map(json.loads, open(O / 'valid64_manifest.jsonl'))}
    RM = {r['pilot_index']: r for r in map(json.loads, open(O / 'reverb16k/manifest.jsonl'))}
    R = {r['sample_id']: r for r in map(json.loads, open(O / 'route_manifest/route_manifest.jsonl'))}
    U = {(u['sample_id'], u['query_delta_deg']): u for u in map(json.loads, open(O / 'route_manifest/direction_uncertainty_manifest.jsonl'))}
    clean_len = {}
    stat = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in sorted(T, key=lambda t: ('clean', 'reverb', 'spatial', 'direction_uncertainty').index(t['condition'])):
        c, i = t['condition'], t['pilot_index']; assert sha(t['audio_path']) == t['audio_sha256']
        w, info = prepare(t); assert w.dtype == np.float32 and w.ndim == 1 and np.isfinite(w).all()
        s = stat[c]
        if c == 'clean':  # = SLURP C1: original 16 kHz, untouched
            x, sr = sf.read(t['audio_path'], dtype='float32'); assert sr == 16000 and t['audio_sha256'] == M[i]['audio_sha256']
            assert np.array_equal(w, x) and info['n_chain8'] == 0; clean_len[i] = len(w)
        elif c == 'reverb':  # = SLURP C1-R: reverb_single.reverb(), 16-bit, used as is
            rm = RM[i]; assert sf.info(t['audio_path']).samplerate == 16000 and sf.info(t['audio_path']).subtype == 'PCM_16'
            assert rm['sha256'] == t['audio_sha256'] and rm['rir_set'] == 'cv_dev93' and rm['rir_pos'] == i % 2
            y, _ = RV.reverb(rm['source'], 'cv_dev93', rm['rir_example'], rm['rir_pos'])
            # reverb_single length rule: 2*len(resample_poly(x,1,2)) = 2*ceil(n/2), i.e. n (even n) or n+1 (odd n).
            # SLURP devel inputs are all even-length; FSC odd-length inputs get one extra sample. Chain not modified.
            n = clean_len[i]; assert len(w) == len(y) == 2 * ((n + 1) // 2) and float(np.abs(w - y).max()) <= Q and info['n_chain8'] == 0
            s['err_vs_reverb_single'].append(float(np.abs(w - y).max())); s['plus1'].append(int(len(w) == n + 1))
        else:  # = SLURP C5: 8 kHz out{k} -> sign once -> chain8 once
            sid = t['sample_id']; r = R[sid]
            u = r if c == 'spatial' else U[(sid, t['query_delta_deg'])]
            assert t['audio_path'] == u['selected_audio_path'] and t['selected_stream'] == u['selected_stream']
            sign = -1.0 if r['flipped'][t['selected_stream']] else 1.0; assert t['polarity_sign'] == sign
            x, sr = sf.read(t['audio_path'], dtype='float64'); assert sr == 8000
            manual = level(resample_poly(sign * x, 2, 1)[:2 * len(x)]).astype(np.float32)  # chain8 written out once
            assert np.array_equal(w, manual) and len(w) == 2 * len(x) and info['n_chain8'] == 1 and info['polarity_applied'] == 1
            # sign applied exactly once: the unsigned version differs whenever sign == -1
            s['sign'].append(sign)
        s['n_chain8'].append(info['n_chain8']); s['len'].append(len(w))
        if c != 'clean':
            s['dbfs'].append(active_dbfs(w))
    res = {}
    for c, s in stat.items():
        res[c] = dict(n=len(s['len']), n_chain8=sorted(set(s['n_chain8'])), len_min=min(s['len']), len_max=max(s['len']))
        if s['dbfs']:
            res[c].update(dbfs_min=min(s['dbfs']), dbfs_max=max(s['dbfs']))
            assert abs(min(s['dbfs']) - LEVEL_DBFS) < 0.05 and abs(max(s['dbfs']) - LEVEL_DBFS) < 0.05, (c, min(s['dbfs']), max(s['dbfs']))
        if s['sign']:
            res[c]['n_sign_minus'] = int(sum(v < 0 for v in s['sign']))
        if s['err_vs_reverb_single']:
            res[c]['max_abs_err_vs_reverb_single'] = max(s['err_vs_reverb_single'])
            res[c]['n_len_plus1_odd_input'] = sum(s['plus1'])
    assert all(res[c]['n'] == 64 for c in ('clean', 'reverb', 'spatial', 'direction_uncertainty'))
    du = collections.Counter(t['query_delta_deg'] for t in T if t['condition'] == 'direction_uncertainty')
    assert du == {-20.0: 16, -10.0: 16, 10.0: 16, 20.0: 16}
    out.update(conditions=res, direction_offsets={str(k): v for k, v in sorted(du.items())}, tasks_sha256=sha(O / 'pilot_tasks.jsonl'),
               reverb_manifest_sha256=sha(O / 'reverb16k/manifest.jsonl'), verdict='PASS')
    (O / 'preprocess_check.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2)); print('PREPROCESS_CHECK_PASS')


if __name__ == '__main__':
    main()
