"""Risk-coverage plot for docs/c6_delta_v1.md (eval half; one line per delta). Reads eval_c6_delta.json only.
Usage (tfgridnet env, has matplotlib): python -B scripts/sms_wsj/plot_c6_risk_coverage.py EVAL_JSON OUT_PNG
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def main(js, png):
    r = json.load(open(js)); tau = r['abstention']['tau_deg']
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for d, pts in r['abstention']['risk_coverage_eval'].items():
        P = [p for p in pts if p['answered']]
        ax.plot([p['coverage'] for p in P], [p['sel_error'] for p in P], marker='.', ms=3, label=f'delta={d}')
        at = [p for p in P if p['tau'] == tau]
        if at:
            ax.plot(at[0]['coverage'], at[0]['sel_error'], 'k+', ms=9)
    ax.set_xlabel('coverage (answered fraction)'); ax.set_ylabel('selection error among answered')
    ax.set_title(f'C6-delta abstention, eval half (+ = tau {tau} deg)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print('WROTE', png)


if __name__ == '__main__':
    main(*sys.argv[1:3])
