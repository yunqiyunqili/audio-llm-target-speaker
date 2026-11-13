"""Risk-coverage plot for docs/c6_real_v1.md (all 200 samples, equal weights, main scoring; abstention A over tau 0..90,
'+' = tau 17, 'x' = abstention B). Reads outputs/c6_real_v1/risk_coverage.json only.
Usage (tfgridnet env, has matplotlib): python -B scripts/sms_wsj/plot_c6_real_risk_coverage.py RC_JSON OUT_PNG
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SHOW = ('sector_K8_center', 'sector_K4_center', 'noise_sigma5', 'noise_sigma10', 'noise_sigma15')
TITLE = dict(spatialnet='SpatialNet ep72 snapshot (diagnosis only)', fasnet_tac='FaSNet-TAC')


def main(js, png):
    r = json.load(open(js)); tau = r['tau_A']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (fe, C) in zip(axes, r['curves'].items()):
        for cn in SHOW:
            P = [p for p in C[cn]['curve'] if p and p['coverage'] > 0]
            ln, = ax.plot([p['coverage'] for p in P], [p['sel_error'] for p in P], marker='.', ms=3, label=cn)
            at = [p for p in P if p['tau'] == tau]
            if at:
                ax.plot(at[0]['coverage'], at[0]['sel_error'], '+', color=ln.get_color(), ms=10, mew=2)
            if C[cn]['B']:
                ax.plot(C[cn]['B']['coverage'], C[cn]['B']['sel_error'], 'x', color=ln.get_color(), ms=9, mew=2)
        ax.set_title(TITLE.get(fe, fe), fontsize=10); ax.set_xlabel('coverage (answered fraction)'); ax.grid(alpha=.3)
    axes[0].set_ylabel('selection error among answered'); axes[0].legend(fontsize=7)
    fig.suptitle(f'Real C6 v1, all 200 samples, indistinguishable = error (+ = margin tau {tau} deg, x = sector rule B)', fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print('WROTE', png)


if __name__ == '__main__':
    main(*sys.argv[1:3])
