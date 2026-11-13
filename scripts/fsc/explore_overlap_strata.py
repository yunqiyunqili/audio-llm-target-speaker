"""POST-HOC EXPLORATORY (not a decision input): FSC extension spatial condition stratified by target/interferer triple overlap.
Read-only. Uses the frozen scoring functions (score_extension_v1.mix_row / predict) unchanged, on the frozen v1 outputs.
Overlap = number of slots (action, object, location) whose labels are equal between target and interferer (labels compared as
given, so 'none' == 'none' counts as equal); also the equal-slot pattern. Per stratum: n samples, C5 (selected stream, equal
weights) target_any and only_interferer for base P2, Audio LoRA, Whisper+T2-LoRA (and original Whisper+T2). No significance tests.
Also: per-observation interferer-hit rates by actual route (correct / wrong) for C5, 8-sector, sigma10 (none), FSC vs SLURP test.
Usage: python -I -B scripts/fsc/explore_overlap_strata.py OUT_JSON
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/fsc'))
import score_extension_v1 as S  # noqa: E402

SLOTS = ('action', 'object', 'location')
ROUTES = ('base', 'audio_lora', 't2_lora', 'wt2')


def main(out):
    I = json.load(open(ROOT / 'outputs/fsc_ext_v1/score_inputs.json')); L = json.load(open(ROOT / I['list']))
    P = {rt: S.load(I['pred']['spatial_selected'][rt]) for rt in ROUTES}
    rows = []
    for g in L['groups']:
        for sw in g['swaps']:
            k = f"g{g['group']:03d}_s{sw['swap']}"; t = sw['target']; i = 'B' if t == 'A' else 'A'
            T, J = g[t]['target'], g[i]['target']
            eq = tuple(s for s in SLOTS if T[s] == J[s])
            r = dict(key=k, n_equal=len(eq), pattern='+'.join(eq) or 'none_equal', stratum=g['rir']['stratum'])
            for rt in ROUTES:
                m = S.mix_row(P[rt][k]['raw_output'], T, J); r[rt] = dict(t_any=m['t_any'], oi=m['cat'] == 'only_interferer')
            rows.append(r)
    def agg(sel):
        n = len(sel)
        return dict(n=n, **{rt: dict(target_any=sum(x[rt]['t_any'] for x in sel) / n, only_interferer=sum(x[rt]['oi'] for x in sel) / n) for rt in ROUTES}) if n else dict(n=0)
    rep = dict(note='POST-HOC EXPLORATORY; not part of the frozen analysis; no significance tests', n=len(rows),
               by_n_equal={str(k): agg([x for x in rows if x['n_equal'] == k]) for k in sorted({x['n_equal'] for x in rows})},
               by_pattern={p: agg([x for x in rows if x['pattern'] == p]) for p in sorted({x['pattern'] for x in rows})},
               by_angle_stratum={s: agg([x for x in rows if x['stratum'] == s]) for s in sorted({x['stratum'] for x in rows})})
    # by actual route, FSC vs SLURP test (counts from the frozen scoring outputs)
    F = json.load(open(ROOT / 'outputs/fsc_ext_v1/score/report.json'))['conditions']['spatial']
    SP = json.load(open(ROOT / 'outputs/slurp_test_v1/score/checks_p2/checks.json'))['conditions']
    ST = json.load(open(ROOT / 'outputs/slurp_test_v1/score/checks_t2/checks.json'))['conditions']
    route = {}
    for cn in ('C5', 'sector_K8_center', 'noise_sigma10'):
        fm = F[cn]['none']['models']
        sm = {'base': SP[cn]['none']['all']['models']['p2'], 'audio_lora': SP[cn]['none']['all']['models']['lora'],
              'wt2': ST[cn]['none']['all']['models']['p2'], 't2_lora': ST[cn]['none']['all']['models']['lora']}
        route[cn] = {}
        for rt in ROUTES:
            for ds, m in (('FSC', fm[rt]['by_actual_route']), ('SLURP', sm[rt]['by_actual_route'])):
                for lab in ('correct', 'wrong'):
                    v = m[lab]; n = v['n_observations']
                    route[cn].setdefault(f'{ds}/{lab}', {})[rt] = dict(n_observations=n, interferer_value_hits=v['interferer_value_hits'],
                                                                      interferer_hits_per_observation=v['interferer_value_hits'] / n if n else None,
                                                                      target_value_hits=v['target_value_hits'])
    rep['by_actual_route_fsc_vs_slurp'] = route
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(rep, indent=1) + '\n')
    f = lambda x: f'{x:.3f}'  # noqa: E731
    for key in ('by_n_equal', 'by_pattern'):
        for s, a in rep[key].items():
            print(key, s, a['n'], ' | '.join(f"{rt} TA {f(a[rt]['target_any'])} OI {f(a[rt]['only_interferer'])}" for rt in ROUTES))
    for cn, d in route.items():
        for lab, v in d.items():
            print(cn, lab, {rt: (x['n_observations'], x['interferer_value_hits'], round(x['interferer_hits_per_observation'], 3)) for rt, x in v.items()})


if __name__ == '__main__':
    main(sys.argv[1])
