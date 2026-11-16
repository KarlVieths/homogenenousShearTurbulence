#!/usr/bin/env python3
"""Lumley invariants and correlation coefficients of the campaign, from the result tables.

   python3 tools/invariants.py            (from doc.production/assets/campaign/)

Reads results/all_modes.csv and results/forced_range.csv and writes
results/invariants.csv.  No run data are needed: the anisotropy tensor
b_ij = <u_i u_j>/q2 - delta_ij/3 in the shear frame (1 = par, 2 = v,
3 = perp) is rebuilt from the tables:

    b11 = b_par, b22 = b_v, b33 = b_perp,
    b12 = <u_par v>/q2 = -uv,   b23 = <u_perp v>/q2 = uperpv,
    b13 = <u_par u_perp>/q2 = rho_pp sqrt((b11 + 1/3)(b33 + 1/3)).

forced_range.csv has no rho_pp, so b13 is set to zero there.  That is
exact by symmetry at theta = 0 and close for the filtered runs (all-modes
|rho_pp| <= 0.023, a change below 0.3 % in II_b), but not for the small box
at theta > 0 (|rho_pp| up to 0.32): those rows are left out.

Invariants (Lumley's convention, and Pope's triangle coordinates):

    II_b  = -1/2 b_ij b_ji,   III_b = 1/3 b_ij b_jk b_ki = det b,
    eta   = sqrt(-II_b/3),    xi    = cbrt(III_b/2).

Correlation coefficients rho_ij = <u_i u_j>/(u_i,rms u_j,rms).

Errors: linear propagation of the tables' block errors of b11, b22, b12,
b13, b23 (b33 = -b11 - b22), treated as independent; the covariances
between components are not in the tables, so the errors are approximate.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
THIRD = 1.0 / 3.0


def tensor(p):
    b11, b22, b12, b13, b23 = p
    b33 = -b11 - b22
    return [[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]]


def quantities(p):
    b = tensor(p)
    b2 = [[sum(b[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tr2 = sum(b2[i][i] for i in range(3))
    tr3 = sum(b2[i][k] * b[k][i] for i in range(3) for k in range(3))
    II, III = -0.5 * tr2, tr3 / 3.0
    eta = math.sqrt(tr2 / 6.0)
    xi = math.copysign(abs(tr3 / 6.0) ** THIRD, tr3)
    var = [b[i][i] + THIRD for i in range(3)]
    rho = lambda i, j: b[i][j] / math.sqrt(var[i] * var[j])
    return {'II_b': II, 'III_b': III, 'xi': xi, 'eta': eta,
            'rho_pv': rho(0, 1), 'rho_pp': rho(0, 2), 'rho_vp': rho(1, 2)}


def propagate(p, s):
    q0 = quantities(p)
    var = {k: 0.0 for k in q0}
    for n in range(len(p)):
        if s[n] == 0.0:
            continue
        h = 1e-6
        pp = list(p); pp[n] += h
        pm = list(p); pm[n] -= h
        qp, qm = quantities(pp), quantities(pm)
        for k in q0:
            var[k] += ((qp[k] - qm[k]) / (2 * h) * s[n]) ** 2
    return q0, {k: math.sqrt(v) for k, v in var.items()}


def rows_of(path, rng):
    out = []
    for r in csv.DictReader(open(os.path.join(RES, path))):
        f = lambda k: float(r[k])
        theta = f('theta')
        b11, b22, b33 = f('b_par'), f('b_v'), f('b_perp')
        if 'rho_pp' in r:
            fac = math.sqrt((b11 + THIRD) * (b33 + THIRD))
            b13, e13 = f('rho_pp') * fac, f('err_rho_pp') * fac
        else:
            if r['case'] == 'box_3x2x1' and theta > 0:
                continue
            b13, e13 = 0.0, 0.0
        p = (b11, b22, -f('uv'), b13, f('uperpv'))
        s = (f('err_b_par'), f('err_b_v'), f('err_uv'), e13, f('err_uperpv'))
        q, e = propagate(p, s)
        if 'rho_pp' in r:          # the tables' own rho's (from the time series) replace the rebuilt ones
            for k in ('rho_pv', 'rho_pp', 'rho_vp'):
                q[k], e[k] = f(k), f('err_' + k)
        else:
            q['rho_pp'], e['rho_pp'] = float('nan'), float('nan')
        out.append((r['case'], theta, rng, q, e))
    return out


KEYS = ['II_b', 'III_b', 'xi', 'eta', 'rho_pv', 'rho_pp', 'rho_vp']


def main():
    rows = rows_of('all_modes.csv', 'all') + rows_of('forced_range.csv', 'forced')
    with open(os.path.join(RES, 'invariants.csv'), 'w') as fo:
        fo.write('case,theta,range,' + ','.join(KEYS) + ',' + ','.join('err_' + k for k in KEYS) + '\n')
        for case, theta, rng, q, e in rows:
            fo.write(f'{case},{theta:g},{rng},' + ','.join(f'{q[k]:.6g}' for k in KEYS) + ','
                     + ','.join(f'{e[k]:.3g}' for k in KEYS) + '\n')
    print(f'{len(rows)} rows -> results/invariants.csv')
    for case, theta, rng, q, e in rows:
        print(f'{case:22s} {theta:5.1f} {rng:6s} ' + '  '.join(f'{k} {q[k]:+.4f}({e[k]:.4f})' for k in KEYS))


if __name__ == '__main__':
    main()
