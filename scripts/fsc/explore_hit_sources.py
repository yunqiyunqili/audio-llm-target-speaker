"""POST-HOC EXPLORATORY (not a decision input): where do FSC interferer-value hits on CORRECTLY routed samples come from?
Read-only; frozen outputs; the frozen mapping is re-traced with provenance by traced_predict(), which is asserted to return exactly
the frozen score_extension_v1.predict() triple for every scored output (so no rule is changed).
Scope: C5 (query = theta1, selected stream), samples whose selected stream == direct-reference target stream (per_sample.jsonl).
(a) every interferer-value hit (predicted slot=value in D_I) attributed to the source of that predicted slot value:
    action    : 'intent' (intent -> action table) or 'object_implied' (bring / change language from an object or language word)
    object    : 'entity' (keyword in a model entity value) or 'intent_default' (intent default object)
    location  : 'entity' (only source; no default location exists)
(b) correct-route samples split into SI-SDRi tertiles (target-stream direct SI-SDRi, per_sample.jsonl 'sisdri_target'):
    interferer hits per observation and only_interferer rate per route.
Usage: python -I -B scripts/fsc/explore_hit_sources.py OUT_JSON
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/fsc'))
import score_extension_v1 as S  # noqa: E402

ROUTES = ('base', 'audio_lora', 'wt2', 't2_lora')
SLOTS = ('action', 'object', 'location')


def traced_predict(raw):
    """Same rules as score_extension_v1.predict(), additionally returning the source of each slot value."""
    obj, st = S.parse(raw)
    if not obj:
        return dict(action='unmapped', object='none', location='none'), dict(action='parse_fail', object='none', location='none')
    vals = [S.toks(S.normalize_v2(v)) for _, v in obj['entities']]
    o = None; src = {}
    for ws in vals:
        hit = [name for name, kw in S.OBJ if ws & kw]
        if hit:
            o = 'lamp' if ('lamp' in hit and 'lights' in hit) else hit[0]; src['object'] = 'entity'; break
    loc = next((name for ws in vals for name, kw in S.LOC if ws & kw), 'none'); src['location'] = 'entity' if loc != 'none' else 'none'
    m = S.I2A.get(obj['intent'], S.I2A['*'])
    if o is None:
        o = m['default_object'] or 'none'; src['object'] = 'intent_default' if o != 'none' else 'none'
    a = m['action']; src['action'] = 'intent'
    if o in S.IMPL:
        a = S.IMPL[o]; src['action'] = 'object_implied'
    elif any(ws & S.LANGW for ws in vals):
        a = 'change language'; src['action'] = 'object_implied'
        if o not in S.IMPL:
            o = 'none'; src['object'] = 'none'
    return dict(action=a, object=o, location=loc), src


def main(out):
    I = json.load(open(ROOT / 'outputs/fsc_ext_v1/score_inputs.json')); L = json.load(open(ROOT / I['list']))
    ps = {r['key']: r for r in map(json.loads, open(ROOT / I['selection_dir'] / 'per_sample.jsonl'))}
    P = {rt: S.load(I['pred']['spatial_selected'][rt]) for rt in ROUTES}
    obs = []
    for g in L['groups']:
        for sw in g['swaps']:
            k = f"g{g['group']:03d}_s{sw['swap']}"
            if not ps[k]['correct']:
                continue
            t = sw['target']; i = 'B' if t == 'A' else 'A'; T, J = g[t]['target'], g[i]['target']
            VT, VI = S.sv(T), S.sv(J); DI = VI - VT
            r = dict(key=k, sisdri=ps[k]['sisdri_target'])
            for rt in ROUTES:
                raw = P[rt][k]['raw_output']; p, src = traced_predict(raw)
                assert p == S.predict(raw)[0], (rt, k)
                hits = [s for s in SLOTS if p[s] != 'none' and DI[f'{s}={p[s]}'] > 0]
                m = S.mix_row(raw, T, J); assert m['di'] == len(hits), (rt, k, m['di'], hits)
                r[rt] = dict(hits=[(s, src[s]) for s in hits], oi=m['cat'] == 'only_interferer')
            obs.append(r)
    n = len(obs)
    rep = dict(note='POST-HOC EXPLORATORY; not part of the frozen analysis; no significance tests', scope='C5 selected stream, correct route only',
               n_correct_route=n, traced_predict_equals_frozen=True, sources={}, by_sisdri_tertile={})
    for rt in ROUTES:
        c = collections.Counter(f'{s}:{src}' for r in obs for s, src in r[rt]['hits'])
        tot = sum(c.values())
        rep['sources'][rt] = dict(total_hits=tot, hits_per_observation=tot / n, by_slot_source=dict(c.most_common()),
                                  from_intent_default=sum(v for k, v in c.items() if k.endswith('intent_default')),
                                  from_intent_action=sum(v for k, v in c.items() if k == 'action:intent'),
                                  from_entity_values=sum(v for k, v in c.items() if k.endswith(':entity') or k.endswith('object_implied')))
    q = np.quantile([r['sisdri'] for r in obs], [1 / 3, 2 / 3])
    tiers = [('low', lambda x: x < q[0]), ('mid', lambda x: q[0] <= x < q[1]), ('high', lambda x: x >= q[1])]
    for name, f in tiers:
        sel = [r for r in obs if f(r['sisdri'])]
        rep['by_sisdri_tertile'][name] = dict(n=len(sel), sisdri_range=[float(min(r['sisdri'] for r in sel)), float(max(r['sisdri'] for r in sel))],
                                              **{rt: dict(hits_per_observation=sum(len(r[rt]['hits']) for r in sel) / len(sel), only_interferer=sum(r[rt]['oi'] for r in sel) / len(sel))
                                                 for rt in ROUTES})
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
