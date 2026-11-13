"""Evaluation split switch for the mix_v1 pipeline (docs/slurp_test_freeze_v1.md §4; added 2026-11-12 for the SLURP test).

Default (env SLURP_EVAL_SPLIT unset or 'devel') returns exactly the devel constants previously hard-coded in the
scripts, so devel behaviour is unchanged. SLURP_EVAL_SPLIT=test switches the mix list, mixture directory and SMS-WSJ RIR
set to the frozen test versions. Nothing here changes any rule, prompt, threshold or scoring function.
"""
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
SH = Path('<SHARED_ROOT>')
SPLIT = os.environ.get('SLURP_EVAL_SPLIT', 'devel')
assert SPLIT in ('devel', 'test', 'fsc_ext'), SPLIT  # fsc_ext: FSC extension v1 (docs/fsc_extension_plan_v1.md), added 2026-11-12
_C = dict(
    devel=dict(LIST=PROJ / 'configs/mix_v1_devel100_v2.json', MIX=SH / 'data/mix_v1/devel_seed2026', RIRSET='cv_dev93',
               DIRECT_REFS=PROJ / 'outputs/frontend_mix_v1_eval_v2'),
    test=dict(LIST=PROJ / 'configs/mix_v1_test_v1.json', MIX=SH / 'data/mix_v1/test_seed2026', RIRSET='test_eval92',
              DIRECT_REFS=PROJ / 'outputs/slurp_test_v1/direct_refs_mix'),
    fsc_ext=dict(LIST=PROJ / 'configs/fsc_ext_v1_list.json', MIX=SH / 'data/fsc_ext_v1/mix', RIRSET='test_eval92',
                 DIRECT_REFS=PROJ / 'outputs/fsc_ext_v1/direct_refs_mix'))[SPLIT]
LIST, MIX, RIRSET, DIRECT_REFS = _C['LIST'], _C['MIX'], _C['RIRSET'], _C['DIRECT_REFS']
MANIFEST = str(MIX / 'manifest.jsonl')
GEOM = PROJ / f'results/sms_wsj_geometry/{RIRSET}_per_mixture.csv'


def natural_counts(rirset):
    """Counts of SMS-WSJ examples per frozen angle stratum ([0,15) [15,30) [30,60) [60,180]) of the speaker separation,
    from results/sms_wsj_geometry/<rirset>_per_mixture.csv (cv_dev93 reproduces the frozen NATURAL = 78/96/155/653)."""
    import csv
    B = [('[0,15)', 0, 15), ('[15,30)', 15, 30), ('[30,60)', 30, 60), ('[60,180]', 60, 180.0001)]
    c = {k: 0 for k, _, _ in B}
    for r in csv.DictReader(open(PROJ / f'results/sms_wsj_geometry/{rirset}_per_mixture.csv', newline='')):
        s = float(r['separation']); c[next(k for k, lo, hi in B if lo <= s < hi)] += 1
    return c


def set_natural(rirset):
    """In-place switch of the frozen NATURAL dict (eval_c6_delta.NATURAL, shared by eval_c6_real / eval_final_lora_checks)
    for the descriptive test_eval92-weighted report (freeze §0-5). cv_dev93 is the frozen main weighting."""
    import eval_c6_delta
    new = natural_counts(rirset)
    if rirset == 'cv_dev93':
        assert new == eval_c6_delta.NATURAL, new
    eval_c6_delta.NATURAL.update(new)
    return dict(eval_c6_delta.NATURAL)
