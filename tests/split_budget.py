#!/usr/bin/env python3
"""Checks of statistics/split_runtime.dat against Runtimedata (selective_lu_orr.md, section 5.3).

    split_budget.py RUNDIR [--tol 2e-3] [--margin 20] [--prev RUNDIR_COARSE --ratio 3] [--switch T1 [T2 ...]]

RUNDIR holds statistics/Runtimedata and statistics/split_runtime.dat of one run of the solver
(a nonlinear run with exact_shift = .true., no large-scale damping, no Stokes layer).  Four checks:

  1. E_tor + E_pol = energy of Runtimedata                          (round-off, 1e-11)
  2. P_LU + P_Orr = -S <uv> - S2 <vw>, the production of Runtimedata (round-off, 1e-11 of its peak)
  3. diss of split_runtime.dat = diss of Runtimedata                (identical expression)
  4. the budget  d/dt (E_tor + E_pol) = sum glu P_LU + sum gorr P_Orr - diss,  dE/dt by central
     differences of the rows: max residual / max |dE/dt| <= tol.

The residual is the error of the discretisation, not of the identity: it is that of the y direction
(it falls by 4 or more for each doubling of ny, not with the time step, which it does not depend on)
plus the O(dt_stat^2) of the central difference; with the default exact_shift = .false. it is dominated
by the O(dy^2) of the shear shift (FINDINGS.md), unweighted or not.  --margin M also requires the budget
without the weights (sum P_LU + sum P_Orr - diss) to miss by M times more than the weighted one, which
is what makes the check sensitive to the weights (void for weights of 1).  With --prev, the residual of
the run of that directory (the coarser ny) divided by this one must be at least --ratio.

With --switch T1 T2 ... (the run has weights that take effect at these times, prod_lu_start and prod_orr_start,
selective_lu_orr.md section 9): the weights jump there, so the rows whose stencil t[n-1] < T < t[n+1] straddles a
switch are left out of check 4 (the central difference averages two rates), and the log of the run (RUNDIR/run.log)
must say that the weights changed at each of these times, to 1e-8 and no other: that is what shows that hst cut its
steps to end on them (a switch time off the step grid).

Only numpy.  Exit status 0 if every check passes.
"""
import argparse
import sys

import numpy as np


def load(rundir):
    split = np.loadtxt(rundir + '/statistics/split_runtime.dat', comments='#')
    rt = np.loadtxt(rundir + '/statistics/Runtimedata')
    if split.shape[0] != rt.shape[0] or np.max(np.abs(split[:, 0] - rt[:, 0])) > 1e-12:
        raise SystemExit('%s: split_runtime.dat and Runtimedata do not have the same rows (an old run in the directory?)' % rundir)
    return split, rt


def logged_switches(rundir):
    """The times of the lines 'production weights from time T: ...' of RUNDIR/run.log (a change of the weights)."""
    import re
    with open(rundir + '/run.log') as f:
        return [float(x) for x in re.findall(r'production weights from time\s+(-?[0-9.]+):', f.read())]


def budget(rundir, switches=()):
    s, rt = load(rundir)
    t, e_tor, e_pol, p_lu, p_orr, gp_lu, gp_orr, eps = s[:, :8].T
    energy, diss, uv, vw, S, S2 = rt[:, 9], rt[:, 10], rt[:, 11], rt[:, 12], rt[:, 3], rt[:, 4]
    prod = -S * uv - S2 * vw
    e = e_tor + e_pol
    de = (e[2:] - e[:-2]) / (t[2:] - t[:-2])
    res = de - (gp_lu + gp_orr - eps)[1:-1]
    res_unw = de - (p_lu + p_orr - eps)[1:-1]
    keep = np.ones(res.size, dtype=bool)
    for T in switches:                       # the stencil of the row straddles the jump of the weights
        keep &= ~((t[:-2] < T) & (T < t[2:]))
    res, res_unw, de_all = res[keep], res_unw[keep], de
    return dict(
        energy=np.max(np.abs(e - energy) / energy),
        prod=np.max(np.abs(p_lu + p_orr - prod)) / np.max(np.abs(prod)),
        diss=np.max(np.abs(eps - diss) / np.maximum(np.abs(diss), 1e-300)),
        res=np.max(np.abs(res)), rel=np.max(np.abs(res)) / np.max(np.abs(de_all)),
        unw=np.max(np.abs(res_unw)), n=len(t), left_out=int((~keep).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir')
    ap.add_argument('--tol', type=float, default=2e-3, help='budget residual relative to max |dE/dt|')
    ap.add_argument('--margin', type=float, default=0.0, help='required (unweighted residual)/(weighted residual)')
    ap.add_argument('--prev', help='run directory at the coarser ny')
    ap.add_argument('--ratio', type=float, default=3.0, help='required residual(prev)/residual(this)')
    ap.add_argument('--switch', type=float, nargs='*', default=[], help='times at which the weights take effect')
    a = ap.parse_args()

    r = budget(a.rundir, a.switch)
    ok = True
    print('   %d rows' % r['n'] + ('; %d left out around the switch times' % r['left_out'] if a.switch else ''))
    if a.switch:
        try:
            got = logged_switches(a.rundir)
        except OSError:
            got = None
        want = sorted(a.switch)
        good = got is not None and len(got) == len(want) and all(abs(g - w) <= 1e-8 for g, w in zip(got, want))
        ok &= good
        print('   %-52s %s   %s' % ('weights changed at (run.log), expected', ', '.join('%.9f' % g for g in got) if got is not None else 'no run.log',
                                    ', '.join('%.9f' % w for w in want) + ('  ok' if good else '  FAIL')))
    for name, val, lim in (('E_tor + E_pol - energy (relative)', r['energy'], 1e-11),
                           ('P_LU + P_Orr + S uv + S2 vw (relative to peak)', r['prod'], 1e-11),
                           ('diss of split_runtime.dat - Runtimedata (relative)', r['diss'], 1e-12)):
        good = val <= lim
        ok &= good
        print('   %-52s %9.2e   limit %.0e  %s' % (name, val, lim, 'ok' if good else 'FAIL'))
    good = r['rel'] <= a.tol
    ok &= good
    print('   %-52s %9.2e   limit %.0e  %s   (absolute %.2e)' % ('budget residual / max |dE/dt|', r['rel'], a.tol, 'ok' if good else 'FAIL', r['res']))
    if a.margin > 0.0:
        m = r['unw'] / r['res']
        good = m >= a.margin
        ok &= good
        print('   %-52s %9.1f   limit %.0f  %s' % ('unweighted budget residual / weighted', m, a.margin, 'ok' if good else 'FAIL'))
    if a.prev:
        rp = budget(a.prev, a.switch)
        q = rp['res'] / r['res']
        good = q >= a.ratio
        ok &= good
        print('   %-52s %9.1f   limit %.1f  %s' % ('residual coarser ny / this ny', q, a.ratio, 'ok' if good else 'FAIL'))
    print(' PASSED' if ok else ' FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
