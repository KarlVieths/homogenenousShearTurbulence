#!/usr/bin/env python3
"""Production-filter runs (PRODUCTION.md): what sits in the unforced range.

For every run with production_runtime.dat (time, ncol, and as ly/2 <.>:
(1-G) uu, vv, ww, (1-G) grad u : grad u, G uv, G wv, (1-G) uw) the
averages over t > --transient, in the frame of the shear (par = along the
shear, perp = across it in the plane):

  E_low/q2      share of the energy in the columns below the cutoff
  eps_low/eps   share of the dissipation there (= nonlinear flux into them / eps)
  P/eps         production -S <G u v> - S2 <G w v> over dissipation (budget: 1)
  uvG/uv        share of the shear stress that produces
  b_low, b_high anisotropy (par, v, perp) of the unforced and of the forced range

    python3 tools/prod.py [--transient 400] [--runs runs]   -> prod.csv
"""
import argparse, glob, os, re
import numpy as np
LY, NU = 2.0, 1e-3


def deckval(path, key):
    m = re.search(r'^\s*' + key + r'\s*=\s*([-0-9.eEdD+]+)', open(path).read(), re.M)
    return float(m.group(1).replace('d', 'e').replace('D', 'e'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transient', type=float, default=400.0)
    ap.add_argument('--runs', default='runs')
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, '*', 'theta_*', 'production_runtime.dat'))):
        d = os.path.dirname(f)
        case, theta = d.split(os.sep)[-2], float(d.split('_')[-1])
        S, S2 = deckval(d + '/hst.in', 'S'), deckval(d + '/hst.in', 's2_amplitude')
        LY = deckval(d + '/hst.in', 'ly')            # 2 except in the tall square boxes
        c, s = S/np.hypot(S, S2), S2/np.hypot(S, S2)
        rt, vr, pf = (np.loadtxt(d + '/' + n) for n in ('Runtimedata', 'variances_runtime.dat', 'production_runtime.dat'))
        n = min(len(rt), len(vr), len(pf)); rt, vr, pf = rt[:n], vr[:n], pf[:n]
        k = rt[:, 0] > a.transient
        if k.sum() < 10:
            k = rt[:, 0] > 0.5*rt[-1, 0]
        m = lambda x: x[k].mean()
        E, eps, uv, vw = m(rt[:, 9]), m(rt[:, 10]), m(rt[:, 11]), m(rt[:, 12])       # ly/2 <.>
        uu, ww, vv, uw = (0.5*m(vr[:, i]) for i in (1, 2, 3, 4))                       # ly <.> -> ly/2 <.>
        uul, vvl, wwl, epsl, uvg, vwg, uwl = (m(pf[:, i]) for i in range(2, 9))

        def frame(a_uu, a_vv, a_ww, a_uw):          # (par, v, perp) variances and their anisotropy
            par = a_uu*c*c + 2*a_uw*c*s + a_ww*s*s
            perp = a_uu*s*s - 2*a_uw*c*s + a_ww*c*c
            q = par + a_vv + perp
            return [par/q - 1/3, a_vv/q - 1/3, perp/q - 1/3]
        bl = frame(uul, vvl, wwl, uwl)
        bh = frame(uu - uul, vv - vvl, ww - wwl, uw - uwl)
        P = -(S*uvg + S2*vwg)
        rows.append([case, theta, rt[-1, 0], int(pf[-1, 1]), 2*E/LY, NU*2*eps/LY, (uul + vvl + wwl)/E, epsl/eps,
                     P/(NU*eps), (c*uvg + s*vwg)/(c*uv + s*vw)] + bl + bh)
    hdr = 'case,theta,t,ncol,q2,eps,E_low_over_q2,eps_low_over_eps,P_over_eps,uvG_over_uv,bl_par,bl_v,bl_perp,bh_par,bh_v,bh_perp'
    with open('prod.csv', 'w') as f:
        f.write(hdr + '\n')
        for r in rows:
            f.write(','.join([r[0]] + [f'{x:.6g}' for x in r[1:]]) + '\n')
    print(f'{"case":16s}{"theta":>6s}{"t":>7s}{"ncol":>5s}{"q2":>8s}{"E_low/q2":>9s}{"epsl/eps":>9s}{"P/eps":>7s}{"uvG/uv":>7s}   b_low (par v perp)     b_high (par v perp)')
    for r in rows:
        print(f'{r[0]:16s}{r[1]:6.1f}{r[2]:7.0f}{r[3]:5d}{r[4]:8.4f}{r[6]:9.3f}{r[7]:9.3f}{r[8]:7.3f}{r[9]:7.3f}   '
              + ' '.join(f'{x:6.3f}' for x in r[10:13]) + '    ' + ' '.join(f'{x:6.3f}' for x in r[13:16]))
    print('prod.csv written')


if __name__ == '__main__':
    main()
