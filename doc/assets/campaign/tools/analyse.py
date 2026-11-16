#!/usr/bin/env python3
"""Statistics of the damping campaign in the frame of the mean shear.

   python3 tools/analyse.py [--transient 400] [--block 50]      (from damping_campaign/)

Every run directory runs/<case>/theta_<deg> (case = n<n>_<mask>) with
output is averaged over t > t_start + transient with the functions of
honeycomb_campaign/tools/analyse.py (same quantities: q2, eps, S*,
-<u_par v>/q2, L_eps, Re_lambda, b_par, b_v, b_perp, <u_perp v>/q2, the
rho's; errors from blocks of --block time units).  The honeycomb rect
runs are read from its results.csv as the case "n1" (the small box).
Writes results.csv and angle_dependence.png (--out; --only <regex> keeps
only the matching cases plus n1, e.g. --only hard --out angle_dependence_hard.png)
and prints, per case, the spread (max - min)/|mean| over theta.
"""
import argparse, csv, glob, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HC = os.path.expanduser('~/Codes/hst/honeycomb_campaign')
sys.path.insert(0, os.path.join(HC, 'tools'))
import analyse as hc        # noqa: E402  (load, stats_series, derived, analyse, QUANT, CORR)

COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#7b61ff']
MARKS = ['o', 's', '^', 'D', 'v', 'P']
KEYS = [k for k, _ in hc.QUANT + hc.CORR]


def small_box_rows():
    rows = []
    path = os.path.join(HC, 'results.csv')
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        if r['lattice'] != 'rect':
            continue
        res = {k: float(r[k]) for k in KEYS}
        err = {k: float(r['err_' + k]) for k in KEYS}
        rows.append(('n1', float(r['theta']), float(r['St_run']), r['done'] == '1', res, err))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--transient', type=float, default=400.0)
    ap.add_argument('--block', type=float, default=50.0)
    ap.add_argument('--only', default='', help='regex on the case name (e.g. hard): draw only these cases')
    ap.add_argument('--out', default='angle_dependence.png')
    a = ap.parse_args()
    rows = small_box_rows()
    for d in sorted(glob.glob(os.path.join(HERE, 'runs', '*', 'theta_*'))):
        if not os.path.exists(os.path.join(d, 'Runtimedata')):
            continue
        # case n<n>_<mask>; a suffix after it (n3_hard_p2: phase-2 angles kept apart on disk) is dropped
        case = re.match(r'(?:n\d+|sq[0-9.]+_ly[0-9.]+)_[a-z]+[0-9.]*(?:_edge[0-9.]+)?', os.path.basename(os.path.dirname(d))).group(0)
        m = re.search(r'^\s*ly\s*=\s*([0-9.]+)', open(os.path.join(d, 'hst.in')).read(), re.M)
        hc.LY = float(m.group(1)) if m else 2.0    # the honeycomb analysis assumes ly = 2
        theta = float(os.path.basename(d).split('_')[1])
        out = hc.analyse(d, theta, a.transient, a.block)
        if out is None:
            print(f'{d}: too short'); continue
        res, err, span, done = out
        rows.append((case, theta, span, done, res, err))
    if a.only:
        rows = [r for r in rows if r[0] == 'n1' or re.search(a.only, r[0])]
    cases = []
    for r in rows:
        if r[0] not in cases:
            cases.append(r[0])
    with open(os.path.join(HERE, 'results.csv'), 'w') as f:
        f.write('case,theta,St_run,done,' + ','.join(KEYS) + ',' + ','.join('err_' + k for k in KEYS) + '\n')
        for case, theta, span, done, res, err in rows:
            f.write(f'{case},{theta},{span:.1f},{int(done)},' + ','.join(f'{res[k]:.6g}' for k in KEYS) + ','
                    + ','.join(f'{err.get(k, float("nan")):.3g}' for k in KEYS) + '\n')
    print(f'{len(rows)} runs (transient {a.transient}, blocks {a.block}); results.csv written')
    print('spread over theta <= 45 deg, (max - min)/|mean|, per case:')
    print(f'{"case":12s} {"n":>3s} ' + ' '.join(f'{k:>7s}' for k in ('q2', 'eps', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v', 'b_perp')))
    for case in cases:
        sel = [r for r in rows if r[0] == case and r[1] <= 45.0]
        if len(sel) < 2:
            continue
        out = []
        for k in ('q2', 'eps', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v', 'b_perp'):
            v = np.array([r[4][k] for r in sel])
            out.append((v.max() - v.min())/abs(v.mean()))
        print(f'{case:12s} {len(sel):3d} ' + ' '.join(f'{x:7.3f}' for x in out))
    for case in cases:
        for r in sorted([r for r in rows if r[0] == case], key=lambda r: r[1]):
            m = r[4]
            print(f'  {case:10s} theta {r[1]:5.1f}  t {r[2]:6.0f}{"" if r[3] else " (running)"}  q2 {m["q2"]:.4f}  S* {m["Sstar"]:.3f}'
                  f'  -upv/q2 {m["uv"]:.4f}  b_par {m["b_par"]:.3f}  b_v {m["b_v"]:.3f}  b_perp {m["b_perp"]:.3f}'
                  f'  rho_pv {m["rho_pv"]:.3f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333',
                         'xtick.color': '#555555', 'ytick.color': '#555555'})

    def panel(ax, k, lab):
        ax.axvspan(-2, 46, color='#eeeeee', zorder=0, lw=0)
        for i, case in enumerate(cases):
            sel = sorted([r for r in rows if r[0] == case], key=lambda r: r[1])
            th = [r[1] for r in sel]; v = [r[4][k] for r in sel]; e = [r[5].get(k, 0.0) for r in sel]
            ax.errorbar(th, v, yerr=e, color=COLS[i % 6], marker=MARKS[i % 6], ms=6, lw=2, capsize=2, label=case,
                        mec='white', mew=1.0, zorder=3)
            for r in sel:
                if not r[3]:
                    ax.plot(r[1], r[4][k], marker=MARKS[i % 6], ms=11, mfc='none', mec=COLS[i % 6], mew=1, zorder=4)
        ax.set_title(lab); ax.grid(True, color='#e3e3e3', lw=0.6)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90]); ax.set_xlim(-4, 94)

    # 10 quantities and the 3 correlation coefficients <u_i u_j>/(u_i,rms u_j,rms) in one 3 x 5 figure
    CORR = [(k, lab.split(' = ')[0] + '$' if ' = ' in lab else lab) for k, lab in hc.CORR]
    fig, axs = plt.subplots(3, 5, figsize=(16, 9.4), sharex=True)
    items = hc.QUANT + CORR
    for ax, (k, lab) in zip(axs.flat, items):
        panel(ax, k, lab)
        if k.startswith('rho') or k in ('b_par', 'b_v', 'b_perp', 'uperpv'):
            ax.axhline(0, color='#999999', lw=0.8, zorder=1)
    for ax in axs.flat[len(items):]:
        ax.set_visible(False)
    for ax in list(axs[2, :len(items) - 10]) + list(axs[1, len(items) - 10:]):
        ax.set_xlabel(r'shear direction $\theta$ [deg]'); ax.tick_params(labelbottom=True)
    axs[0, 0].legend(loc='best', frameon=False)
    fig.suptitle('Damping campaign: shear-frame statistics vs shear direction, enlarged box n with the small box\'s '
                 f'ellipse masked (n1 = the small box; averages after S t = {a.transient:g}; open ring: in progress);\n'
                 r'bottom row: correlation coefficients $\rho_{ij} = \langle u_i u_j\rangle/(u_{i,rms}\, u_{j,rms})$ '
                 r'for $(\parallel, v)$, $(\parallel, \perp)$, $(v, \perp)$')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, a.out + '.tmp.png'), dpi=120)
    os.replace(os.path.join(HERE, a.out + '.tmp.png'), os.path.join(HERE, a.out))
    print(f'{a.out} written (with the correlation coefficients)')


if __name__ == '__main__':
    main()
