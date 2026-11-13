"""FSC pilot single-speaker reverb (= SLURP C1-R chain), CPU, login node.

Calls scripts/lora/reverb_single.reverb() unchanged. reverb() reads `SLURP / file`; an absolute FSC path makes the
pathlib join resolve to the FSC file itself, so no SLURP code is modified. RIR: the cv_dev93 example of the mix_v1
group used for this pilot pair (configs/mix_v1_devel100_v2.json groups[:32], same as spatial_list.json), position
h_{pilot_index % 2} (A -> h_0, B -> h_1, as in swap 1), mic 0. Written as 16-bit PCM like reverb_single.py.
The previous (incorrect) reverb16k/ is moved to reverb16k_void_20261109/ before writing; nothing is deleted.
"""
import hashlib, json, sys
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lora'))
import reverb_single as RV  # noqa: E402

FSC = ROOT / 'data/fsc/source/fluent_speech_commands_dataset'; O = ROOT / 'outputs/fsc_pilot'
OUT = O / 'reverb16k'; VOID = O / 'reverb16k_void_20261109'


def main():
    M = [json.loads(x) for x in open(O / 'valid64_manifest.jsonl')]; assert [m['pilot_index'] for m in M] == list(range(64))
    L = json.load(open(ROOT / 'configs/mix_v1_devel100_v2.json'))['groups'][:32]
    SP = json.load(open(O / 'spatial_list.json'))['groups']
    assert not VOID.exists(); OUT.rename(VOID); OUT.mkdir()
    rows = []
    for m in M:
        i = m['pilot_index']; g = L[i // 2]; assert SP[i // 2]['rir'] == g['rir']
        rd = Path(g['rir']['rir_dir']); assert rd.parent.name == 'cv_dev93' and rd.parent == RV.RIRS / 'cv_dev93'
        src = (FSC / m['path']).resolve(); assert hashlib.sha256(src.read_bytes()).hexdigest() == m['audio_sha256']
        y, info = RV.reverb(str(src), 'cv_dev93', rd.name, i % 2)
        p = OUT / f'{i:03d}.wav'; sf.write(p, y, 16000, subtype='PCM_16')
        rows.append(dict(pilot_index=i, source=str(src), rir_set='cv_dev93', rir_example=rd.name, rir_pos=i % 2, **info,
                         wav=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with open(OUT / 'manifest.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True) + '\n')
    print(json.dumps(dict(n=len(rows), clipped_total=sum(r['clipped'] for r in rows), peak_max=max(r['peak'] for r in rows),
                          manifest_sha256=hashlib.sha256((OUT / 'manifest.jsonl').read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
