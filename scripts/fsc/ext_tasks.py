"""FSC extension v1 task lists (after construction and selection). CPU.
clean / c0 : original FSC 16 kHz (c0 also carries the gold transcription as text input); reverb: ext_make_reverb output;
spatial_selected / spatial_other : SpatialNet out{stream}.wav (8 kHz) of the v3 selection / other stream, polarity sign from
flipped[stream], preprocess chain8 (scripts/fsc/pilot_audio.prepare, condition 'spatial').
Usage: python -I scripts/fsc/ext_tasks.py LIST_JSON REVERB_DIR SEP_DIR SELECTION_DIR OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FSC = ROOT / 'data/fsc/source/fluent_speech_commands_dataset'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(lst, rev, sep, seld, out):
    L = json.load(open(lst)); rev, sep, seld, out = map(Path, (rev, sep, seld, out)); out.mkdir(parents=True, exist_ok=False)
    RM = {r['item_id']: r for r in map(json.loads, open(rev / 'manifest.jsonl'))}
    T = dict(clean=[], reverb=[], c0=[])
    for it in L['single']:
        base = dict(item_id=it['item_id'], text=it['text'], target=it['target'])
        T['clean'].append(dict(base, condition='clean', audio_path=str(FSC / it['path']), audio_sha256=sha(FSC / it['path'])))
        p = rev / RM[it['item_id']]['wav']; assert sha(p) == RM[it['item_id']]['sha256']
        T['reverb'].append(dict(base, condition='reverb', audio_path=str(p), audio_sha256=RM[it['item_id']]['sha256']))
        T['c0'].append(dict(base, condition='c0'))
    G = {g['group']: g for g in L['groups']}
    for role, fn in (('selected', 'selection.jsonl'), ('other', 'other_selection.jsonl')):
        T[f'spatial_{role}'] = []
        for s in map(json.loads, open(seld / fn)):
            k = s['key']; g = G[int(k[1:4])]; sw = g['swaps'][int(k[-1]) - 1]; t = sw['target']; i = 'B' if t == 'A' else 'A'
            p = sep / k / f"out{s['stream']}.wav"
            T[f'spatial_{role}'].append(dict(item_id=k, condition='spatial', stream_role=role, stream=s['stream'], preprocess='chain8',
                                             polarity_sign=-1.0 if s['flipped'][s['stream']] else 1.0, audio_path=str(p), audio_sha256=sha(p),
                                             text=g[t]['sentence'], target=g[t]['target'], interferer=g[i]['target']))
    for k, v in T.items():
        (out / f'{k}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in v))
        print(k, len(v), sha(out / f'{k}.jsonl'), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
