"""FSC extension v1 reverb condition (= SLURP C1-R chain, test_eval92): scripts/lora/reverb_single.reverb() unchanged, RIR
assignment from configs/fsc_ext_v1_list.json (single[].reverb). 16-bit PCM like reverb_single.py. CPU.
Usage: python -I scripts/fsc/ext_make_reverb.py LIST_JSON OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import soundfile as sf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lora'))
import reverb_single as RV  # noqa: E402
FSC = ROOT / 'data/fsc/source/fluent_speech_commands_dataset'


def main(lst, out):
    L = json.load(open(lst)); out = Path(out); out.mkdir(parents=True, exist_ok=False); rows = []
    for it in L['single']:
        r = it['reverb']; y, info = RV.reverb(str((FSC / it['path']).resolve()), r['rir_set'], r['rir_example'], r['rir_pos'])
        p = out / f"{it['item_id']}.wav"; sf.write(p, y, 16000, subtype='PCM_16')
        rows.append(dict(item_id=it['item_id'], **r, **info, wav=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in rows))
    print(json.dumps(dict(n=len(rows), clipped_total=sum(r['clipped'] for r in rows), peak_max=max(r['peak'] for r in rows))), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
