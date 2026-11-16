#!/usr/bin/env python3
"""Smooth-edge production filter (n = 5) against the unfiltered 3:2:1 box, 0..45 deg -> smooth_vs_small_box.png
(reads results.csv as written by `tools/analyse.py --only n5_prod`)."""
import csv, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = [r for r in csv.DictReader(open('results.csv')) if float(r['theta']) <= 45]
CASES = [('n1', 'unfiltered 3:2:1 box (3 x 2 x 1)', '#2a78d6', 'o'),
         ('n5_prod3.63', 'filter, sharp cutoff at 3.63', '#eb6834', 's'),
         ('n5_prod3.80', 'filter, sharp cutoff at 3.80', '#eda100', 'D'),
         ('n5_prod3.63_edge0.3', 'filter, smooth cutoff at 3.63', '#1baf7a', '^')]
Q = [('q2', r'$q^2$'), ('eps', r'$\varepsilon$'), ('Sstar', r'$S^* = S q^2/\varepsilon$'), ('uv', r'$-\langle u_\parallel v\rangle/q^2$'),
     ('Leps', r'$L_\varepsilon$'), ('Relam', r'$Re_\lambda$'), ('b_par', r'$b_{\parallel\parallel}$'), ('b_v', r'$b_{vv}$'),
     ('b_perp', r'$b_{\perp\perp}$'), ('uperpv', r'$\langle u_\perp v\rangle/q^2$')]
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333', 'xtick.color': '#555555', 'ytick.color': '#555555'})
fig, axs = plt.subplots(2, 5, figsize=(17, 6.6), sharex=True)
for ax, (k, lab) in zip(axs.ravel(), Q):
    for c, name, col, mk in CASES:
        sel = sorted([r for r in rows if r['case'] == c], key=lambda r: float(r['theta']))
        ax.errorbar([float(r['theta']) for r in sel], [float(r[k]) for r in sel], yerr=[float(r['err_' + k]) for r in sel],
                    color=col, marker=mk, ms=7, lw=2, capsize=2, mec='white', mew=1, label=name)
    ax.set_title(lab); ax.grid(True, color='#e3e3e3', lw=0.6); ax.set_xticks([0, 15, 30, 45])
    if k in ('b_perp', 'uperpv'): ax.axhline(0, color='#bbbbbb', lw=1, zorder=0)
for ax in axs[1]: ax.set_xlabel(r'shear direction $\theta$ [deg]')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle(r'Production filter (box 15 x 2 x 5, $k_c$ = 3.63 = $2\pi/\sqrt{3}$ or 3.80, Re = 1000) against the unfiltered 3:2:1 box: shear-frame statistics vs shear direction (averages over S t = 400..1400, block errors)', y=0.995, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig('smooth_vs_small_box.png', dpi=130); print('smooth_vs_small_box.png')
