#!/usr/bin/env python3
"""Statistics of the honeycomb campaign in the frame of the mean shear.

   python3 tools/analyse.py [--transient 200] [--block 50]     (from honeycomb_campaign/)

For every run directory with output: time averages over t > t_start +
transient of Runtimedata and variances_runtime.dat (hst_stats.f90 for the
columns; all lab-frame second moments are there), rotated into the shear
frame e_par = (cos theta, sin theta) in (x, z), e_perp = (-sin, cos):

  q2, eps = nu <du_i/dx_j du_i/dx_j>, S* = S q2/eps, P/eps,
  -<u_par v>/q2, the anisotropies b_par, b_v, b_perp, b_par,v,
  L_eps = (q2/3)^(3/2)/eps, Re_lambda = (q2/3) sqrt(15/(nu eps)),
  and the off-axis stress <u_perp v>/q2 (zero by symmetry only when theta is
  a symmetry direction of the lattice), and the correlation coefficients
  rho_pv, rho_pp, rho_vp = <u_i u_j>/(u_i,rms u_j,rms) for (par, v),
  (par, perp), (v, perp) (the last two vanish by symmetry at 0 and 90 deg).

Error bars: standard error from block averages of length --block (in S t).
Writes results.csv, angle_dependence.png, correlations.png and prints the spread over
theta <= 45 per lattice.  Runs still in progress are included (marked).
"""
import argparse
import glob
import io
import math
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LY, NU, S0 = 2.0, 1.0e-3, 1.0
LATTICES = [('rect', 'rectangle 3 x 1', '#2a78d6', 'o'),
            ('hex3', 'hexagon, Lx/Lz = 3', '#eb6834', 's'),
            ('hex4.3', 'hexagon, Lx/Lz = 4.3', '#1baf7a', '^'),
            ('sq', 'square 1 x 1 (1:2:1)', '#eda100', 'D'),
            ('sq3', 'square sqrt3 x sqrt3 (area 3)', '#e87ba4', 'v')]
QUANT = [('q2', r'$q^2$'), ('eps', r'$\varepsilon$'), ('Sstar', r'$S^* = S q^2/\varepsilon$'),
         ('uv', r'$-\langle u_\parallel v\rangle/q^2$'), ('Leps', r'$L_\varepsilon$'),
         ('Relam', r'$Re_\lambda$'), ('b_par', r'$b_{\parallel\parallel}$'), ('b_v', r'$b_{vv}$'),
         ('b_perp', r'$b_{\perp\perp}$'), ('uperpv', r'$\langle u_\perp v\rangle/q^2$')]
CORR = [('rho_pv', r'$\rho_{\parallel v} = \langle u_\parallel v\rangle/(u_{\parallel,rms}\, v_{rms})$'),
        ('rho_pp', r'$\rho_{\parallel\perp} = \langle u_\parallel u_\perp\rangle/(u_{\parallel,rms}\, u_{\perp,rms})$'),
        ('rho_vp', r'$\rho_{v\perp} = \langle v\, u_\perp\rangle/(v_{rms}\, u_{\perp,rms})$')]


def loadtxt(path):
    # Fortran writes |x| < 1e-99 as 1.234-100 (no E): put the E back (slow,
    # only when the plain read fails)
    try:
        return np.loadtxt(path, ndmin=2)
    except ValueError:
        pass
    with open(path) as f:
        text = re.sub(r'(\d)([+-]\d{3})(?=\s|$)', r'\1E\2', f.read())
    return np.loadtxt(io.StringIO(text), ndmin=2)


def load(d):
    rt = loadtxt(os.path.join(d, 'Runtimedata'))
    var = loadtxt(os.path.join(d, 'variances_runtime.dat'))
    n = min(len(rt), len(var))
    rt, var = rt[:n], var[:n]
    # a restarted run appends from its restart time: keep the last record of
    # every time (scan backwards, keep strictly decreasing times)
    keep = np.zeros(n, bool); tmin = np.inf
    for i in range(n - 1, -1, -1):
        if rt[i, 0] < tmin - 1e-9:
            keep[i] = True; tmin = rt[i, 0]
    return rt[keep], var[keep]


def stats_series(rt, var, theta):
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    q2 = 2*rt[:, 9]/LY
    eps = NU*2*rt[:, 10]/LY
    uv, wv = 2*rt[:, 11]/LY, 2*rt[:, 12]/LY
    uu, ww, vv, uw = var[:, 1]/LY, var[:, 2]/LY, var[:, 3]/LY, var[:, 4]/LY
    upar_v = c*uv + s*wv
    uperp_v = -s*uv + c*wv
    upar2 = c*c*uu + 2*c*s*uw + s*s*ww
    uperp2 = s*s*uu - 2*c*s*uw + c*c*ww
    upar_uperp = c*s*(ww - uu) + (c*c - s*s)*uw
    return dict(t=rt[:, 0], q2=q2, eps=eps, upar_v=upar_v, uperp_v=uperp_v, upar2=upar2, uperp2=uperp2, vv=vv,
                upar_uperp=upar_uperp)


def derived(m):
    q2, eps = m['q2'], m['eps']
    return dict(q2=q2, eps=eps, Sstar=S0*q2/eps, uv=-m['upar_v']/q2, Peps=-m['upar_v']*S0/eps,
                Leps=(q2/3)**1.5/eps, Relam=(q2/3)*math.sqrt(15/(NU*eps)),
                b_par=m['upar2']/q2 - 1/3, b_v=m['vv']/q2 - 1/3, b_perp=m['uperp2']/q2 - 1/3,
                uperpv=m['uperp_v']/q2,
                rho_pv=m['upar_v']/np.sqrt(m['upar2']*m['vv']),
                rho_pp=m['upar_uperp']/np.sqrt(m['upar2']*m['uperp2']),
                rho_vp=m['uperp_v']/np.sqrt(m['vv']*m['uperp2']))


def analyse(d, theta, transient, block):
    rt, var = load(d)
    ser = stats_series(rt, var, theta)
    t = ser['t']
    t0 = t[0] + transient
    keep = t > t0
    if keep.sum() < 10:
        return None
    nb = int((t[-1] - t0)//block)
    means = {k: ser[k][keep].mean() for k in ('q2', 'eps', 'upar_v', 'uperp_v', 'upar2', 'uperp2', 'vv', 'upar_uperp')}
    res = derived(means)
    err = {}
    if nb >= 3:
        blocks = []
        for b in range(nb):
            sel = (t > t0 + b*block) & (t <= t0 + (b + 1)*block)
            blocks.append(derived({k: ser[k][sel].mean() for k in means}))
        for k in res:
            v = np.array([bl[k] for bl in blocks])
            err[k] = v.std(ddof=1)/math.sqrt(nb)
    return res, err, t[-1] - t[0], os.path.exists(os.path.join(d, 'DONE'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transient', type=float, default=200.0)
    ap.add_argument('--block', type=float, default=50.0)
    a = ap.parse_args()
    rows = []
    for key, label, col, mk in LATTICES:
        for d in sorted(glob.glob(os.path.join(HERE, 'runs', key, 'theta_*'))):
            if not os.path.exists(os.path.join(d, 'Runtimedata')):
                continue
            theta = float(os.path.basename(d).split('_')[1])
            r = analyse(d, theta, a.transient, a.block)
            if r is None:
                continue
            res, err, span, done = r
            rows.append((key, theta, span, done, res, err))
    if not rows:
        print('no run has passed the transient yet')
        return
    keys = list(rows[0][4].keys())
    with open(os.path.join(HERE, 'results.csv.tmp'), 'w') as f:
        f.write('lattice,theta,St_run,done,' + ','.join(keys) + ',' + ','.join('err_' + k for k in keys) + '\n')
        for key, theta, span, done, res, err in rows:
            f.write(f'{key},{theta},{span:.1f},{int(done)},' + ','.join(f'{res[k]:.6g}' for k in keys) + ','
                    + ','.join(f'{err.get(k, float("nan")):.3g}' for k in keys) + '\n')
    os.replace(os.path.join(HERE, 'results.csv.tmp'), os.path.join(HERE, 'results.csv'))
    print(f'{len(rows)} runs (transient {a.transient}, blocks {a.block}); results.csv written')
    print('spread over theta <= 45 deg, (max - min)/mean, per lattice:')
    print(f'{"lattice":8s} {"n":>3s} ' + ' '.join(f'{k:>7s}' for k in ('q2', 'eps', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v')))
    for key, label, col, mk in LATTICES:
        sel = [r for r in rows if r[0] == key and r[1] <= 45.0]
        if len(sel) < 2:
            continue
        out = []
        for k in ('q2', 'eps', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v'):
            v = np.array([r[4][k] for r in sel])
            out.append((v.max() - v.min())/abs(v.mean()))
        print(f'{key:8s} {len(sel):3d} ' + ' '.join(f'{x:7.3f}' for x in out))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333',
                         'xtick.color': '#555555', 'ytick.color': '#555555'})
    def panel(ax, k, lab):
        ax.axvspan(-2, 46, color='#eeeeee', zorder=0, lw=0)
        for key, label, col, mk in LATTICES:
            sel = sorted([r for r in rows if r[0] == key], key=lambda r: r[1])
            if not sel:
                continue
            th = [r[1] for r in sel]
            v = [r[4][k] for r in sel]
            e = [r[5].get(k, 0.0) for r in sel]
            ax.errorbar(th, v, yerr=e, color=col, marker=mk, ms=6, lw=2, capsize=2, label=label,
                        mec='white', mew=1.0, zorder=3)
            for r in sel:
                if not r[3]:
                    ax.plot(r[1], r[4][k], marker=mk, ms=11, mfc='none', mec=col, mew=1, zorder=4)
        ax.set_title(lab)
        ax.grid(True, color='#e3e3e3', lw=0.6)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_xlim(-4, 94)

    fig, axs = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
    for ax, (k, lab) in zip(axs.flat, QUANT):
        panel(ax, k, lab)
    for ax in axs[1]:
        ax.set_xlabel(r'shear direction $\theta$ [deg]')
    axs[0, 0].legend(loc='best', frameon=False)
    fig.suptitle('Honeycomb campaign: statistics in the shear frame vs. shear direction '
                 f'(averages after S t = {a.transient:g}; grey band: theta <= 45; open ring: run in progress)')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'angle_dependence.png.tmp'), dpi=120, format='png')
    os.replace(os.path.join(HERE, 'angle_dependence.png.tmp'), os.path.join(HERE, 'angle_dependence.png'))
    print('angle_dependence.png written')

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (k, lab) in zip(axs, CORR):
        panel(ax, k, lab)
        ax.axhline(0, color='#999999', lw=0.8, zorder=1)
        ax.set_xlabel(r'shear direction $\theta$ [deg]')
    axs[0].legend(loc='best', frameon=False)
    fig.suptitle(r'Correlation coefficients $\langle u_i u_j\rangle/(u_{i,rms} u_{j,rms})$ in the shear frame '
                 r'($e_\parallel = (\cos\theta, \sin\theta)$, $e_\perp = (-\sin\theta, \cos\theta)$ in (x, z); $v$ along the gradient)'
                 f' vs. shear direction (after S t = {a.transient:g}; grey band: theta <= 45)')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'correlations.png.tmp'), dpi=120, format='png')
    os.replace(os.path.join(HERE, 'correlations.png.tmp'), os.path.join(HERE, 'correlations.png'))
    print('correlations.png written')


if __name__ == '__main__':
    main()
