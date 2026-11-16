#!/usr/bin/env python3
"""Shear-frame statistics of the forced range only (k_h >= the forcing shell) -> highband.csv, highband_vs_small_box.png

Filter runs: total (Runtimedata, variances_runtime.dat) minus the unforced share that the code
integrates at run time with the weight 1 - G (production_runtime.dat); for the smooth edge the
"forced range" is therefore G-weighted, for the sharp one it is k_h >= k_c exactly.
Unfiltered 3:2:1 box (honeycomb rect): the only columns with k_h < k_c = 2 pi/sqrt 3 are (0,0),
which the run-time sums do not contain, and (+-2 pi/3, 0); the latter's variances, stresses and dissipation come from the velocity snapshots
(t >= 400) and are subtracted from the time averages.
"""
import glob, math, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
HC = os.path.expanduser('~/Codes/hst/honeycomb_campaign'); sys.path.insert(0, os.path.join(HC, 'tools'))
import analyse as hc
NU, T0, BLOCK = 1e-3, 400.0, 50.0
KEYS = ['q2', 'eps', 'Sstar', 'uv', 'Leps', 'Relam', 'b_par', 'b_v', 'b_perp', 'uperpv']

def lab2frame(th, uu, vv, ww, uw, uv, wv, q2, eps):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return dict(q2=q2, eps=eps, upar_v=c*uv + s*wv, uperp_v=-s*uv + c*wv, upar2=c*c*uu + 2*c*s*uw + s*s*ww,
                uperp2=s*s*uu - 2*c*s*uw + c*c*ww, vv=vv, upar_uperp=c*s*(ww - uu) + (c*c - s*s)*uw)

def blocks(t, ser):
    nb = int((t[-1] - T0)//BLOCK); out = []
    for b in range(nb):
        k = (t > T0 + b*BLOCK) & (t <= T0 + (b + 1)*BLOCK); out.append({q: v[k].mean() for q, v in ser.items()})
    return out

def filter_case(d, th):
    hc.LY = LY = float(re.search(r'^\s*ly\s*=\s*([0-9.]+)', open(d + '/hst.in').read(), re.M).group(1))
    rt, var = hc.load(d); pf = np.loadtxt(d + '/production_runtime.dat'); n = min(len(rt), len(pf)); rt, var, pf = rt[:n], var[:n], pf[:n]
    assert abs(rt[:, 0] - pf[:, 0]).max() < 1e-6
    f = 2/LY; uul, vvl, wwl, epsl, uvg, vwg, uwl = (f*pf[:, i] for i in range(2, 9))
    uu, ww, vv, uw = (var[:, i]/LY for i in (1, 2, 3, 4))
    ser = lab2frame(th, uu - uul, vv - vvl, ww - wwl, uw - uwl, uvg, vwg, f*rt[:, 9] - uul - vvl - wwl, NU*(f*rt[:, 10] - epsl))
    bl = blocks(rt[:, 0], ser); mean = {q: np.mean([b[q] for b in bl]) for q in ser}
    der = hc.derived(mean); per = [hc.derived(b) for b in bl]
    return der, {k: np.std([p[k] for p in per], ddof=1)/math.sqrt(len(per)) for k in KEYS}, 0.0

def small_box(d, th):
    hc.LY = LY = 2.0
    rt, var = hc.load(d); f = 2/LY
    tot = dict(uu=var[:, 1]/LY, ww=var[:, 2]/LY, vv=var[:, 3]/LY, uw=var[:, 4]/LY, uv=f*rt[:, 11], wv=f*rt[:, 12], q2=f*rt[:, 9], eps=NU*f*rt[:, 10])
    bl = blocks(rt[:, 0], tot); low = []; chk = []; chk0 = []
    for fn in sorted(glob.glob(d + '/fields/field*.fld'), key=lambda s: int(re.findall(r'field(\d+)', s)[0])):
        head = open(fn, 'rb').read(600).decode('latin1'); t = float(re.search(r'\bt_field=([-0-9.eE+]+)', head).group(1))
        if t < T0: continue
        nx, nzs, nyc = (int(re.search(k + r'=(\d+)', head).group(1)) for k in ('nx', 'ny', 'nz'))
        a0 = float(re.search(r'alpha0=([-0-9.eE+]+)', head).group(1))
        shape = (nx + 1, 2*nzs + 1, nyc + 3, 3)
        V = np.memmap(fn, dtype='<c16', mode='r', offset=os.path.getsize(fn) - 16*int(np.prod(shape)), shape=shape)
        dy = LY/(shape[2] - 4); acc = dict.fromkeys(('uu', 'vv', 'ww', 'uw', 'uv', 'wv', 'eps'), 0.0)
        for ix, wgt in ((1, 2.0),):                                # (2 pi/3, 0) with its conjugate; the run-time sums hold no (0,0) column (checked below)
            col = np.array(V[ix, nzs, :, :]); u, w, v = (col[2:-2, i] for i in range(3)); kh2 = (a0*ix)**2
            g = (col[3:-1, :] - col[1:-3, :])/(2*dy)                 # d/dy with the ghost rows
            acc['uu'] += wgt*(abs(u)**2).mean(); acc['vv'] += wgt*(abs(v)**2).mean(); acc['ww'] += wgt*(abs(w)**2).mean()
            acc['uw'] += wgt*(u*np.conj(w)).real.mean(); acc['uv'] += wgt*(u*np.conj(v)).real.mean(); acc['wv'] += wgt*(w*np.conj(v)).real.mean()
            acc['eps'] += wgt*NU*(kh2*(abs(col[2:-2])**2).sum(-1).mean() + (abs(g)**2).sum(-1).mean())
        acc['q2'] = acc['uu'] + acc['vv'] + acc['ww']; low.append(acc)
        m2 = np.ones(nx + 1)*2; m2[0] = 1
        e = (abs(np.array(V[:, :, 2:-2, :]))**2).mean(2).sum(-1)*m2[:, None]; i0 = np.abs(rt[:, 0] - t).argmin()
        chk.append((e.sum() - e[0, nzs])/tot['q2'][i0]); chk0.append(e.sum()/tot['q2'][i0])
    lm = {q: np.mean([l[q] for l in low]) for q in low[0]}; le = {q: np.std([l[q] for l in low], ddof=1)/math.sqrt(len(low)) for q in low[0]}
    hi = lambda b, l: hc.derived(lab2frame(th, *(b[q] - l[q] for q in ('uu', 'vv', 'ww', 'uw', 'uv', 'wv', 'q2', 'eps'))))
    mean = {q: np.mean([b[q] for b in bl]) for q in tot}; der = hi(mean, lm)
    per = [hi(b, lm) for b in bl]; ps = [hi(mean, l) for l in low]      # scatter of the totals (blocks) and of the removed part (snapshots)
    err = {k: math.hypot(np.std([p[k] for p in per], ddof=1)/math.sqrt(len(per)), np.std([p[k] for p in ps], ddof=1)/math.sqrt(len(ps))) for k in KEYS}
    print('  n1 theta %4.1f: %d snapshots, snapshot q2/Runtimedata q2 without (0,0) %.3f..%.3f, with %.3f..%.3f; removed: q2 %.1f +- %.1f %%, vv %.1f %%, eps %.1f %%, -uv %.1f %%'
          % (th, len(low), min(chk), max(chk), min(chk0), max(chk0), 100*lm['q2']/mean['q2'], 100*le['q2']/mean['q2'], 100*lm['vv']/mean['vv'], 100*lm['eps']/mean['eps'], 100*lm['uv']/mean['uv']))
    return der, err, lm['q2']/mean['q2']

CASES = [('n1', 'unfiltered 3:2:1 box (3 x 2 x 1)', '#2a78d6', 'o'), ('n5_prod3.63', 'filter, sharp cutoff at 3.63', '#eb6834', 's'),
         ('n5_prod3.80', 'filter, sharp cutoff at 3.80', '#eda100', 'D'),
         ('n5_prod3.63_edge0.3', 'filter, smooth cutoff at 3.63', '#1baf7a', '^')]
rows = []
for c, *_ in CASES:
    ds = sorted(glob.glob(HC + '/runs/rect/theta_*') if c == 'n1' else glob.glob('runs/%s/theta_*' % c))
    for d in ds:
        th = float(d.split('_')[-1])
        if th > 45: continue
        der, err, lowfrac = (small_box if c == 'n1' else filter_case)(d, th); rows.append((c, th, der, err))
        print('%-22s %4.1f ' % (c, th) + ' '.join('%s %.4g(%.2g)' % (k, der[k], err[k]) for k in KEYS), flush=True)
with open('highband.csv', 'w') as f:
    f.write('case,theta,' + ','.join(KEYS) + ',' + ','.join('err_' + k for k in KEYS) + '\n')
    for c, th, der, err in rows: f.write('%s,%g,' % (c, th) + ','.join('%.6g' % der[k] for k in KEYS) + ',' + ','.join('%.3g' % err[k] for k in KEYS) + '\n')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
Q = [('q2', r'$q^2$'), ('eps', r'$\varepsilon$'), ('Sstar', r'$S^* = S q^2/\varepsilon$'), ('uv', r'$-\langle u_\parallel v\rangle/q^2$'),
     ('Leps', r'$L_\varepsilon$'), ('Relam', r'$Re_\lambda$'), ('b_par', r'$b_{\parallel\parallel}$'), ('b_v', r'$b_{vv}$'),
     ('b_perp', r'$b_{\perp\perp}$'), ('uperpv', r'$\langle u_\perp v\rangle/q^2$')]
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333', 'xtick.color': '#555555', 'ytick.color': '#555555'})
fig, axs = plt.subplots(2, 5, figsize=(17, 6.6), sharex=True)
for ax, (k, lab) in zip(axs.ravel(), Q):
    for c, name, col, mk in CASES:
        sel = sorted([r for r in rows if r[0] == c], key=lambda r: r[1])
        ax.errorbar([r[1] for r in sel], [r[2][k] for r in sel], yerr=[r[3][k] for r in sel], color=col, marker=mk, ms=7, lw=2, capsize=2, mec='white', mew=1, label=name)
    ax.set_title(lab); ax.grid(True, color='#e3e3e3', lw=0.6); ax.set_xticks([0, 15, 30, 45])
    if k in ('b_perp', 'uperpv'): ax.axhline(0, color='#bbbbbb', lw=1, zorder=0)
for ax in axs[1]: ax.set_xlabel(r'shear direction $\theta$ [deg]')
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle(r'Forced range only ($k_h \geq k_c$; filter runs in the box 15 x 2 x 5): shear-frame statistics vs shear direction (S t = 400..end, errors: blocks and snapshots)', y=0.995, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig('highband_vs_small_box.png', dpi=130); print('highband_vs_small_box.png')
