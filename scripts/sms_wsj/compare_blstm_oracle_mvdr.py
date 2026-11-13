"""Per-utterance comparison of existing test scores (no inference): BLSTM+MVDR vs
oracle power-IRM MVDR vs cACGMM+MVDR, all on direct-path mic0 targets, test_eval92.
Reads only existing jsonl files; geometry from results/sms_wsj_geometry/test_eval92_per_mixture.csv."""
import csv, json, statistics as st, sys
from pathlib import Path
MC = Path('<SEPARATION_REPO>/multichannel')
SRC = {'blstm_mvdr': MC/'blstm_mvdr/exp/sms_blstm_mvdr_v1/test_utterances.jsonl',
       'oracle_mvdr': MC/'oracle_mvdr/exp/21977751/test/utterances.jsonl',
       'cacgmm_mvdr': MC/'spatial_baseline/exp/full_21976118/test/utterances.jsonl'}
out = Path(sys.argv[1])
rows = {k: {r['id']: r for r in map(json.loads, p.read_text().splitlines())} for k, p in SRC.items()}
ids = sorted(rows['blstm_mvdr'])
assert all(set(v) == set(ids) for v in rows.values()), 'id sets differ'
geo = {r['id']: r for r in csv.DictReader(open('results/sms_wsj_geometry/test_eval92_per_mixture.csv'))}
mix_diff = max(abs(rows[k][i]['mixture_si_sdr_db'] - rows['blstm_mvdr'][i]['mixture_si_sdr_db']) for k in rows for i in ids)
def m(x): return round(st.mean(x), 4)
rep = dict(sources={k: str(v) for k, v in SRC.items()}, utterances=len(ids),
           max_abs_mixture_si_sdr_diff_between_files=mix_diff,
           mean={k: {s: m([rows[k][i][s] for i in ids]) for s in ('si_sdr_db', 'si_sdri_db', 'mixture_si_sdr_db')} for k in rows})
d = [rows['blstm_mvdr'][i]['si_sdr_db'] - rows['oracle_mvdr'][i]['si_sdr_db'] for i in ids]
rep['blstm_minus_oracle_si_sdr'] = dict(mean=m(d), median=round(st.median(d), 4), stdev=round(st.stdev(d), 4),
    frac_blstm_better=round(sum(x > 0 for x in d)/len(d), 4), min=round(min(d), 4), max=round(max(d), 4))
d2 = [rows['blstm_mvdr'][i]['si_sdr_db'] - rows['cacgmm_mvdr'][i]['si_sdr_db'] for i in ids]
rep['blstm_minus_cacgmm_si_sdr'] = dict(mean=m(d2), frac_blstm_better=round(sum(x > 0 for x in d2)/len(d2), 4))
rep['permutation_agreement_blstm_oracle'] = round(sum(rows['blstm_mvdr'][i]['permutation'] == rows['oracle_mvdr'][i]['permutation'] for i in ids)/len(ids), 4)
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    return round(sum((x-ma)*(y-mb) for x, y in zip(a, b)) / (sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5, 4)
rep['pearson_si_sdr_blstm_vs_oracle'] = pearson([rows['blstm_mvdr'][i]['si_sdr_db'] for i in ids], [rows['oracle_mvdr'][i]['si_sdr_db'] for i in ids])
by = {}
for name, key, edges in [('separation_deg', 'separation', [0, 15, 30, 60, 180.001]), ('t60_s', 't60', [0.2, 0.3, 0.4, 0.501])]:
    by[name] = {}
    for lo, hi in zip(edges, edges[1:]):
        sel = [i for i in ids if lo <= float(geo[i][key]) < hi]
        by[name][f'[{lo},{round(hi,3)})'] = dict(n=len(sel), **{k: m([rows[k][i]['si_sdri_db'] for i in sel]) for k in rows})
rep['si_sdri_by_bucket'] = by
(out/'comparison.json').write_text(json.dumps(rep, indent=2)+'\n')
print(json.dumps(rep, indent=1))
