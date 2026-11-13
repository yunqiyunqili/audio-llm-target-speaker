"""CPU-only FSC acquisition audit and frozen valid pilot manifest."""
import csv, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FSC=ROOT/'data/fsc/source/fluent_speech_commands_dataset'
OUT=ROOT/'outputs/fsc_pilot'
OUT.mkdir(parents=True,exist_ok=True)
norm=lambda s: re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s.lower())).strip()
def load(name): return list(csv.DictReader(open(FSC/'data'/name)))
splits={k:load(k+'_data.csv') for k in ('train','valid','test')}
slurp=[json.loads(x) for x in open(ROOT/'data/slurp/annotations/8eb1654/train.jsonl')]
sl_text=Counter(norm(x['sentence']) for x in slurp)
audit={'source':'FSC archive (obtain from the official Fluent.ai channel)','archive_sha256':hashlib.sha256((ROOT/'data/fsc/downloads/fluent.tar.gz').read_bytes()).hexdigest(),'splits':{},'overlap':{}}
for name,rs in splits.items():
    texts=[norm(r['transcription']) for r in rs]
    audit['splits'][name]={'n':len(rs),'speakers':len({r['speakerId'] for r in rs}),'speaker_ids':sorted({r['speakerId'] for r in rs}),'intent_triples':len({(r['action'],r['object'],r['location']) for r in rs}),'exact_transcript_overlap_slurp_train':sum(sl_text[t] for t in texts),'unique_exact_overlap_values':sorted({t for t in texts if t in sl_text})}
rng=random.Random(2026); valid=list(splits['valid']); rng.shuffle(valid)
by=defaultdict(list)
for r in valid: by[(r['action'],r['object'],r['location'])].append(r)
chosen=[]; used=set(); counts=Counter()
for passno in range(2):
    for key in sorted(by):
        cand=[r for r in sorted(by[key],key=lambda r:(counts[r['speakerId']],r['speakerId'],r['path'])) if r['path'] not in used]
        if cand:
            chosen.append(cand[0]); used.add(cand[0]['path']); counts[cand[0]['speakerId']]+=1
for r in sorted(valid,key=lambda r:(counts[r['speakerId']],r['speakerId'],r['path'])):
    if len(chosen)>=64: break
    if r not in chosen: chosen.append(r); counts[r['speakerId']]+=1
assert len(chosen)==64 and len({r['speakerId'] for r in chosen})>=8
manifest=[]
for i,r in enumerate(sorted(chosen,key=lambda r:r['path'])):
    p=FSC/r['path']
    manifest.append({'pilot_index':i,'split':'valid','path':r['path'],'speaker_id':r['speakerId'],'transcript':r['transcription'],'target':{'action':r['action'],'object':r['object'],'location':r['location']},'audio_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'valid64_manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in manifest))
audit['pilot']={'n':64,'seed':2026,'speakers':len({x['speaker_id'] for x in manifest}),'speaker_counts':dict(Counter(x['speaker_id'] for x in manifest)),'intent_counts':{ '|'.join(x):n for x,n in Counter(tuple(y['target'].values()) for y in manifest).items()},'manifest_sha256':hashlib.sha256((OUT/'valid64_manifest.jsonl').read_bytes()).hexdigest(),'test_split_used':False}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
config={'status':'frozen_cpu_precheck_complete','dataset':'FSC','source':'FSC (official Fluent.ai channel)','split':'valid','n':64,'seed':2026,'manifest':'outputs/fsc_pilot/valid64_manifest.jsonl','models':{'qwen_base':'frozen_existing','audio_lora':'step1/lr1e-4/step440','whisper_t2_lora':'step2/lr1e-4/step660'},'conditions':['clean','reverb','spatial_two_speaker','doa_uncertainty'],'test_split_used':False,'slurp_final_test_used':False,'scoring':'fsc_exact_slots_plus_frozen_binding_metrics','gpu_budget_hours':2}
(OUT/'pilot_config.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(audit['pilot'],indent=2)); print('exact_overlap_valid',audit['splits']['valid']['exact_transcript_overlap_slurp_train'])
