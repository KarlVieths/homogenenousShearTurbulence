#!/usr/bin/env python3
"""Figures and band tables of doc.production/production_filter.md: the unfiltered 3 x 2 x 1 box, the sharp filter at
k_c = 3.80 and the smooth filter at k_c = 3.63 (both 15 x 2 x 5), all at theta = 0 (par = x, perp = z).

   python3 tools/doc_figures.py  -> doc_spectra_maps.png, doc_spectra_1d.png, doc_timeseries.png, tables on stdout

Spectra from the velocity snapshots (compspec/*.npz, tools/compspec_extract.py; S t >= 600 for the filter runs,
>= 400 for the small box), as densities so that the lattices compare directly.
"""
import math, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
HC = os.path.expanduser('~/Codes/hst/honeycomb_campaign'); sys.path.insert(0, os.path.join(HC, 'tools'))
import analyse as hc
KC = 2*math.pi/math.sqrt(3); KMAX = 16.0
CASES = [('n1', 'unfiltered 3 x 2 x 1 box', '#2a78d6', None, 0.0, HC + '/runs/rect/theta_00.0'),
         ('n5_prod3.80', 'sharp filter, $k_c$ = 3.80', '#eda100', 3.8, 0.0, 'runs/n5_prod3.80/theta_00.0'),
         ('n5_prod3.63_edge0.3', 'smooth filter, $k_c$ = 3.63', '#1baf7a', KC, 0.3, 'runs/n5_prod3.63_edge0.3/theta_00.0')]
def load(name):
    d = np.load(f'compspec/{name}_theta_00.0.npz'); nx1, nz2 = d['uu'].shape; nz = (nz2 - 1)//2; a0, b0 = float(d['alfa0']), float(d['beta0'])
    full = lambda a: np.concatenate([a[:0:-1, ::-1], a], axis=0)          # kx = -nx..nx, using E(-k) = E(k)
    R = dict(a0=a0, b0=b0, kx=a0*np.arange(-(nx1 - 1), nx1), kz=b0*(np.arange(nz2) - nz), n=len(d['times']),
             E=[full(d[k]) for k in ('uu', 'vv', 'ww')] + [-full(d['uv'])])
    KX, KZ = np.meshgrid(R['kx'], R['kz'], indexing='ij'); R['kh'] = np.hypot(KX, KZ); R['KX'] = KX; R['KZ'] = KZ
    for e in R['E']: e[R['kh'] < 1e-9] = 0                                 # the (0,0) column is not in the campaign's statistics
    return R
runs = [(c, load(c[0])) for c in CASES]
names = [r'$\langle u_\parallel^2\rangle$', r'$\langle v^2\rangle$', r'$\langle u_\perp^2\rangle$', r'$-\langle u_\parallel v\rangle$']

# ---- maps
fig, axs = plt.subplots(3, 4, figsize=(17, 13.2))
for r, (c, R) in enumerate(runs):
    ok = R['kh'] <= KMAX + max(R['a0'], R['b0'])
    vx = R['KX'][ok][:, None] + 0.5*R['a0']*np.array([-1, 1, 1, -1]); vz = R['KZ'][ok][:, None] + 0.5*R['b0']*np.array([-1, -1, 1, 1])
    for i in range(4):
        ax = axs[r, i]; ax.set_aspect('equal'); ax.set_xlim(-KMAX*1.13, KMAX*1.13); ax.set_ylim(-KMAX*1.13, KMAX*1.13); ax.axis('off')
        e = R['E'][i]/(R['a0']*R['b0'])
        pc = PolyCollection(np.stack([vx, vz], -1), array=np.maximum(e[ok], 1e-12), cmap='viridis', norm=LogNorm(3e-6, 3e-3), linewidths=0)
        pc.set_clip_path(Circle((0, 0), KMAX, transform=ax.transData)); ax.add_collection(pc)
        for rr in (4, 8, 12, 16):
            ax.add_patch(Circle((0, 0), rr, fill=False, lw=0.35 if rr < KMAX else 0.9, color='0.2')); ax.text(rr*math.cos(0.38), rr*math.sin(0.38), str(rr), fontsize=7, color='w', ha='center', va='center')
        if c[3]:
            ax.add_patch(Circle((0, 0), c[3], fill=False, lw=1.2, color='k'))
            if c[4]:
                for f in (1 - c[4], 1 + c[4]): ax.add_patch(Circle((0, 0), c[3]*f, fill=False, lw=0.8, ls='--', color='k'))
        ax.text(KMAX*1.03, 0, r'$k_x$', fontsize=11, va='center'); ax.text(0, KMAX*1.03, r'$k_z$', fontsize=11, ha='center', va='bottom')
        if r == 0: ax.set_title(names[i], fontsize=14, pad=16)
        if i == 0: ax.text(-0.06, 0.5, c[1], transform=ax.transAxes, rotation=90, va='center', fontsize=12)
fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.02, hspace=0.08, wspace=0.05)
fig.colorbar(pc, cax=fig.add_axes([0.94, 0.25, 0.01, 0.5]), label=r'density per unit area of the $k_x$-$k_z$ plane')
fig.suptitle(r'$\theta$ = 0: variance and shear-stress density per $(k_x, k_z)$ column (rings: $k_h$ = 4, 8, 12, 16; black circle: $k_c$, dashed: the edge of the smooth filter)', fontsize=12)
fig.savefig('doc_spectra_maps.png', dpi=100)

# ---- one-dimensional premultiplied spectra
fig, axs = plt.subplots(3, 4, figsize=(17, 11), sharex='row')
for r, (c, R) in enumerate(runs):
    for i in range(4):
        e = R['E'][i]; st = dict(color=c[2], marker='o', ms=5 if r == 0 else 2.5, lw=1.3, label=c[1])
        kz = R['kz'][R['kz'] > 0]; ez = (e[:, R['kz'] > 0].sum(0) + e[:, R['kz'] < 0].sum(0)[::-1])/R['b0']
        kx = R['kx'][R['kx'] > 0]; ex = (e[R['kx'] > 0].sum(1) + e[R['kx'] < 0].sum(1)[::-1])/R['a0']
        axs[0, i].semilogx(kz, kz*ez, **st); axs[1, i].semilogx(kx, kx*ex, **st)
        edges = np.logspace(math.log10(0.4), math.log10(200), 28) if r else np.logspace(math.log10(1.5), math.log10(200), 16)
        h = np.histogram(R['kh'], bins=edges, weights=e)[0]; kc_ = np.sqrt(edges[1:]*edges[:-1]); m = h != 0
        axs[2, i].semilogx(kc_[m], (h/np.log(edges[1:]/edges[:-1]))[m], **st)
        if r == 0: axs[0, i].set_title(names[i], fontsize=14)
for a in axs.ravel():
    a.grid(lw=0.3, alpha=0.5); a.axvline(KC, color='#1baf7a', lw=0.9, ls='--'); a.axvline(3.8, color='#eda100', lw=0.9, ls='--'); a.axhline(0, color='0.6', lw=0.6)
for a, l in zip(axs[:, 0], [r'$k_\perp E(k_\perp)$', r'$k_\parallel E(k_\parallel)$', r'$k_h E(k_h)$']): a.set_ylabel(l, fontsize=12)
for row, l in zip(axs, [r'$k_\perp = k_z$ (summed over $k_x$)', r'$k_\parallel = k_x$ (summed over $k_z$)', r'$k_h$ (rings)']):
    for b in row: b.set_xlabel(l)
axs[0, 0].legend(fontsize=9, frameon=False)
fig.suptitle(r'$\theta$ = 0: premultiplied spectra (area under a curve on the log axis = contribution to the variance or stress; dashed: the two $k_c$; rows $k$ = 0 are in the tables)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig('doc_spectra_1d.png', dpi=100)

# ---- band tables (shares of each quantity, snapshots)
print('snapshots:', [(c[0], R['n']) for c, R in runs])
def share(R, i, m): return R['E'][i][m].sum()/R['E'][i].sum()
BANDS = [('k_h < 3.63', lambda R: R['kh'] < KC - 1e-6), ('3.63 <= k_h < 6.28', lambda R: (R['kh'] >= KC - 1e-6) & (R['kh'] < 2*math.pi - 1e-6)),
         ('6.28 <= k_h < 12.6', lambda R: (R['kh'] >= 2*math.pi - 1e-6) & (R['kh'] < 4*math.pi - 1e-6)), ('k_h >= 12.6', lambda R: R['kh'] >= 4*math.pi - 1e-6),
         ('row k_perp = 0 (rollers)', lambda R: abs(R['KZ']) < 1e-9), ('row k_par = 0 (infinitely long)', lambda R: abs(R['KX']) < 1e-9),
         ('|k_par| < 2 pi/3, |k_perp| >= 2 pi', lambda R: (abs(R['KX']) < 2*math.pi/3 - 1e-6) & (abs(R['KZ']) >= 2*math.pi - 1e-6))]
for i, nm in enumerate(('uu', 'vv', 'ww', '-uv')):
    print('share of', nm)
    for b, f in BANDS: print('  %-36s' % b + '  '.join('%6.3f' % share(R, i, f(R)) for c, R in runs))
print('share of q2')
for b, f in BANDS:
    print('  %-36s' % b + '  '.join('%6.3f' % (sum(R['E'][i][f(R)].sum() for i in range(3))/sum(R['E'][i].sum() for i in range(3))) for c, R in runs))
print('snapshot totals uu vv ww -uv:'); [print('  ', c[0], ' '.join('%.4f' % R['E'][i].sum() for i in range(4))) for c, R in runs]

# ---- time series of q2 and of the production
fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
print('time series, S t >= 400: mean q2, rms/mean of q2, of -<uv>, of eps; integral time scale of q2')
for ax, (c, R) in zip(axs, runs):
    hc.LY = 2.0; rt, var = hc.load(c[5]); t = rt[:, 0]; q2 = rt[:, 9]; uv = -rt[:, 11]; eps = rt[:, 10]; k = t >= 400
    ax.plot(t, q2, color=c[2], lw=0.8); ax.set_ylabel(r'$q^2$'); ax.set_ylim(0, 0.33); ax.grid(lw=0.3, alpha=0.5)
    ax.text(0.01, 0.86, c[1], transform=ax.transAxes, fontsize=11)
    x = q2[k] - q2[k].mean(); dt = np.median(np.diff(t[k])); n = int(200/dt)
    ac = np.array([np.mean(x[:len(x) - j]*x[j:]) for j in range(0, n, max(1, n//400))])/x.var(); lag = np.arange(0, n, max(1, n//400))*dt
    j0 = np.argmax(ac < 0) if (ac < 0).any() else len(ac); T = np.trapz(ac[:j0], lag[:j0])
    print('  %-22s q2 %.4f  rms/mean: q2 %.3f  -uv %.3f  eps %.3f   T_int %.1f' % (c[0], q2[k].mean(), q2[k].std()/q2[k].mean(), uv[k].std()/uv[k].mean(), eps[k].std()/eps[k].mean(), T))
axs[-1].set_xlabel('S t'); axs[-1].set_xlim(0, 1400)
fig.suptitle(r'$\theta$ = 0: box-averaged $q^2$ (in units of $(S L_z)^2$ of the small box) against time', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig('doc_timeseries.png', dpi=100)
print('written')
