"""CPU-only: complete Audio LoRA − Whisper+T2-LoRA paired comparison on C1 / C1-R for selection, report and all halves.

Rule identical to the reported-half comparison (docs/lora_v2_runs.md §15 item 4): binding-score-v1 target_any with empty
interferer set, bootstrap grouped by slurp_id, B=2000, np.random.default_rng(2026); hit()/boot()/rows() are imported
unchanged from compare_audio_vs_t2.py. Existing report-half numbers must reproduce exactly, otherwise the script stops.
Adds keys to comparison.json; the pre-existing file is first copied to comparison_before_halves.json (existing keys unchanged).
"""
import hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lora'))
from compare_audio_vs_t2 import rows, hit, boot  # noqa: E402

AE = ROOT / 'outputs/lora_audio_eval/lr1e-4'
TE = ROOT / 'outputs/lora_step2_eval/lr1e-4/step660'
SRC = {
    ('C1', 'selection'): (AE / '22025280/step440/devel_clean_selection/raw.jsonl', TE / 'devel_clean_selection_22052673/T2/raw.jsonl'),
    ('C1', 'report'): (AE / '22027861/step440/devel_clean_report/raw.jsonl', TE / 'devel_clean_report_22053122/T2/raw.jsonl'),
    ('C1R', 'selection'): (AE / '22027862/step440/devel_rev_selection/raw.jsonl', TE / 'devel_rev_all_22053123/T2/raw.jsonl'),
    ('C1R', 'report'): (AE / '22027863/step440/devel_rev_report/raw.jsonl', TE / 'devel_rev_all_22053123/T2/raw.jsonl'),
}
OUT = ROOT / 'outputs/lora_step2_eval/audio_vs_t2_comparison_v2/comparison.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(cond, half):
    ap, tp = SRC[(cond, half)]
    A, T = rows(ap), rows(tp)
    T = {k: v for k, v in T.items() if k in A}
    assert set(A) == set(T) and len(A) == 214, (cond, half, len(A), len(T))
    for k in A:  # same recording, same gold
        assert A[k]['slurp_id'] == T[k]['slurp_id'] and A[k]['ref_entities'] == T[k]['ref_entities'], k
    return A, T


def entry(A, T, inputs):
    d = {k: (A[k]['slurp_id'], hit(A[k]) - hit(T[k])) for k in A}
    return {'n': len(A), 'n_slurp_id': len({v[0] for v in d.values()}),
            'audio_any': sum(hit(x) for x in A.values()) / len(A), 't2_any': sum(hit(x) for x in T.values()) / len(T),
            'paired_audio_minus_t2': boot(d), 'inputs': inputs}


def main():
    old = json.loads(OUT.read_text())
    res = {}
    for cond in ('C1', 'C1R'):
        H = {h: load(cond, h) for h in ('selection', 'report')}
        assert not set(H['selection'][0]) & set(H['report'][0])
        assert not {r['slurp_id'] for r in H['selection'][0].values()} & {r['slurp_id'] for r in H['report'][0].values()}
        for h in ('selection', 'report'):
            res[f'{cond}_{h}'] = entry(*H[h], {'audio': str(SRC[(cond, h)][0].relative_to(ROOT)), 'audio_sha256': sha(SRC[(cond, h)][0]),
                                               't2': str(SRC[(cond, h)][1].relative_to(ROOT)), 't2_sha256': sha(SRC[(cond, h)][1])})
        A = {**H['selection'][0], **H['report'][0]}; T = {**H['selection'][1], **H['report'][1]}
        res[f'{cond}_all'] = entry(A, T, 'union of selection and report inputs above')
        assert res[f'{cond}_all']['n'] == 428 and res[f'{cond}_all']['n_slurp_id'] == 382
        # existing reported-half result must reproduce exactly
        o = old[f'{cond}_report']; n = res[f'{cond}_report']
        assert o['n'] == n['n'] and o['audio_any'] == n['audio_any'] and o['t2_any'] == n['t2_any'] and o['paired_audio_minus_t2'] == n['paired_audio_minus_t2'], cond
    backup = OUT.with_name('comparison_before_halves.json')
    assert not backup.exists(); shutil.copy2(OUT, backup)
    new = dict(old)
    for k in ('C1_selection', 'C1_all', 'C1R_selection', 'C1R_all'):
        new[k] = res[k]
    new['halves_note'] = ('selection/all added 2026-11-11 by scripts/lora/compare_audio_vs_t2_halves.py with the same rule; '
                          'report-half entries unchanged and reproduced exactly; external claims use report half only; '
                          'selection-half intervals are optimistic because both checkpoints were selected on C1 selection half.')
    new['before_halves_sha256'] = sha(backup)
    OUT.write_text(json.dumps(new, indent=2, ensure_ascii=False) + '\n')
    for k in ('C1_selection', 'C1_report', 'C1_all', 'C1R_selection', 'C1R_report', 'C1R_all'):
        r = res[k]; p = r['paired_audio_minus_t2']
        print(f"{k:14s} n={r['n']} audio={r['audio_any']:.4f} t2={r['t2_any']:.4f} delta={p['delta']:+.4f} CI=[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]")
    print('SHA256', sha(OUT)); print('HALVES_DONE')


if __name__ == '__main__':
    main()
