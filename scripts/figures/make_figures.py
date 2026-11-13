"""Three core figures for the README / report (docs/figures/). Reads existing result files only; computes nothing new.
  fig1_error_ladder.png  : SLURP test target-any along C0 -> C1 -> C4-direct -> C5 -> C3, original vs LoRA
  fig2_risk_coverage.png : SLURP test c6-real-v1 main conditions, abstention-A risk-coverage (selection error vs coverage,
                           equal weights; route independent), tau = 17 deg and abstention B marked
  fig3_forest.png        : Audio LoRA - Whisper+T2-LoRA, SLURP devel (report half, 95% only), SLURP test and FSC (95% and 98.33%)
Usage (tfgridnet env has matplotlib): python -B scripts/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'docs/figures'; OUT.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(ROOT / p))  # noqa: E731
T = J('outputs/slurp_test_v1/score/report/report.json')
F = J('outputs/fsc_ext_v1/score/report.json')
RC = J('outputs/slurp_test_v1/score/real_base/risk_coverage.json')
DEVC = J('outputs/lora_step2_eval/audio_vs_t2_comparison_v2/comparison.json')
DEV5 = J('outputs/lora_step2_eval/audio_vs_t2_final_checks_corrected/checks.json')


def fig1():
    s, m = T['single'], T['mix']
    stages = ['C0\n(gold text)', 'C1\n(clean)', 'C4-direct\n(not run on test)', 'C5\n(SpatialNet+route)', 'C3\n(mixture+direction text)']
    base = [s['C0']['T2']['any_gold_value_hit'], s['C1']['P2']['any_gold_value_hit'], None, m['C5']['base']['P2']['target_any'], m['C3']['p2']['target_any']]
    lora = [s['C0']['T2LORA']['any_gold_value_hit'], s['C1']['AUDIO']['any_gold_value_hit'], None, m['C5']['audio_lora']['target_any'], m['C3']['audio_lora']['target_any']]
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); x = range(len(stages))
    for ys, lab, mk in ((base, 'original Qwen (C0: T2; others: P2)', 'o'), (lora, 'LoRA (C0: T2-LoRA; others: audio LoRA)', 's')):
        pts = [(i, y) for i, y in zip(x, ys) if y is not None]
        ax.plot(list(x), [float('nan') if y is None else y for y in ys], marker=mk, label=lab)  # gap: no line through C4-direct
        for i, y in pts:
            ax.annotate(f'{y:.3f}', (i, y), textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.axvspan(1.6, 2.4, color='0.92'); ax.text(2, 0.5, 'not run\non SLURP test', ha='center', fontsize=8, color='0.4')
    ax.set_xticks(list(x)); ax.set_xticklabels(stages, fontsize=8); ax.set_ylim(0, 1); ax.set_ylabel('target-any')
    ax.set_title('SLURP test: error decomposition ladder (target-any)', fontsize=10)
    ax.text(0.01, 0.02, 'C0/C1: single-talker (all gold values); C5/C3: two-talker (distinguishable values only)', transform=ax.transAxes, fontsize=7, color='0.35')
    ax.legend(fontsize=8, loc='upper right'); fig.tight_layout(); fig.savefig(OUT / 'fig1_error_ladder.png', dpi=160); plt.close(fig)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (cn, title) in zip(axes, (('sector_K8_center', '8-sector query (centre)'), ('noise_sigma10', 'sigma = 10 deg query noise (x10)'))):
        C = RC['curves']['spatialnet'][cn]; cur = C['curve']
        ax.plot([c['coverage'] for c in cur], [c['sel_error'] for c in cur], '-', label='abstention A (margin >= tau), tau = 0..90 deg')
        p = next(c for c in cur if c['tau'] == RC['tau_A'])
        ax.plot(p['coverage'], p['sel_error'], 'o', color='C1', label=f"tau = {RC['tau_A']} deg (frozen)")
        if C['B']:
            ax.plot(C['B']['coverage'], C['B']['sel_error'], 's', color='C2', label='abstention B (sector)')
        ax.set_title(title, fontsize=9); ax.set_xlabel('coverage (equal weights)'); ax.set_xlim(0, 1.02); ax.legend(fontsize=7)
    axes[0].set_ylabel('selection error among answered\n(indistinguishable counted as error)')
    fig.suptitle('SLURP test, c6-real-v1 main conditions: risk-coverage (route independent)', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig2_risk_coverage.png', dpi=160); plt.close(fig)


def fig3():
    rows = []
    d5 = DEV5['conditions']['C5']['none']['report']['paired']['target_any']['eq']
    for lab, v in (('C1', DEVC['C1_report']['paired_audio_minus_t2']), ('C1-R', DEVC['C1R_report']['paired_audio_minus_t2']), ('C5', d5)):
        rows.append((f'SLURP devel (report half) {lab}', v['delta'], v['ci95'], None))
    for lab, k in (('C1', 'C1'), ('C1-R', 'C1R'), ('C5', 'C5')):
        v = T['primary'][k]; rows.append((f'SLURP test {lab}', v['delta'], v['ci95'], v['ci9833']))
    for lab, k in (('clean', 'clean'), ('reverb', 'reverb'), ('spatial (C5)', 'spatial')):
        v = F['primary'][k]; rows.append((f'FSC {lab}', v['delta'], v['ci95'], v['ci9833']))
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); n = len(rows)
    for i, (lab, d, c95, c98) in enumerate(rows):
        y = n - 1 - i
        if c98:
            ax.plot(c98, [y, y], color='0.6', lw=5, solid_capstyle='butt', label='98.33% (Bonferroni, 3 comparisons)' if i == 3 else None)
        ax.plot(c95, [y, y], color='C0', lw=1.6, label='95%' if i == 0 else None)
        ax.plot(d, y, 'o', color='k', ms=4)
    ax.axvline(0, color='r', lw=0.8, ls='--'); ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    for yb in (2.5, 5.5):
        ax.axhline(yb, color='0.85', lw=0.8)
    ax.set_xlabel('Audio LoRA - Whisper+T2-LoRA, target-any (paired)')
    ax.set_title('Primary comparison; devel had no Bonferroni (95% only)', fontsize=10); ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(OUT / 'fig3_forest.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    fig1(); fig2(); fig3()
    for p in sorted(OUT.glob('*.png')):
        print(p.relative_to(ROOT))
