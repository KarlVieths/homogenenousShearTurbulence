#!/usr/bin/env python3
"""The full matrix of the Kelvin-mode test in an oscillating S2 (tests/KELVIN_S2.md), on demand.

    tests/kelvin_s2_study.py [--build build-cpu] [--jobs 4] [--quick] [--out study.md]

For every case it writes a deck, runs build/test_kelvin_s2 with refine = 2 (deltat,
deltat/2, deltat/4), checks the finest run's table with kelvin_ref.py and prints
  - the errors in v, eta, energy and the projection residual at each deltat,
  - the observed order in time,
and for every oscillating case compares it with the constant-S2 control of the same
mode and G (run over a shorter time, since a constant S2 tilts the mode without bound).

Matrix (specification section 4.1): modes (a) kx = 0 and (b) kx = 2 pi/15, both kz = 2 pi/5,
ky0 = pi; G = 0, 0.547 (smooth edge) and 1; A/S = 1 with S T = 1.75, 5.25, 14; nu = 1e-3
and (nearly) 0; box 15 x 2 x 5 reduced to the one column.  (a) runs 10 periods; (b) as many
periods as keep max|ky| dy below 0.1.  deltat = min(0.1, T/50), then halved twice.

Pass criteria: error <= 1e-5 at the finest deltat, projection residual <= 5e-8, gamma_y to
1e-10, observed order of v and eta >= 1.6 (while the error is above 5e-7: the long runs reach
|ky| dy = 0.1, where the spatial error of the stencils is about 1e-6 and flattens the energy),
and the oscillating order of v within 0.4 of the control's.  The order of the energy is shown only.
The time error is third order for nu = 0 and second order with nu = 1e-3 (Crank-Nicolson).
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import kelvin_ref as K   # noqa: E402

KZ = 2 * np.pi / 5
KX = {'a': 0.0, 'b': 2 * np.pi / 15}
NY = 256
DY = 2.0 / NY
KC = {('a', '0'): 2.0, ('b', '0'): 2.0, ('a', 'h'): 1.2332, ('b', 'h'): 1.2998, ('a', '1'): 0.0, ('b', '1'): 0.0}


def duration(mode, T, const):
    """Time to run: 10 periods for (a); for (b) and the constant control, until max|ky| dy = 0.1."""
    if mode == 'a' and not const:
        return 10 * T
    t = np.linspace(0, 200, 40001)
    m = K.Mode(KX[mode], KZ, np.pi, 1.0, 1.0, 1.0, T if not const else 0.0, 0.0, 1.0, 0.0)
    over = np.nonzero(np.abs(m.ky(t)) * DY > 0.1)[0]
    tend = t[over[0]] if over.size else 200.0
    if const:
        return max(1.0, np.floor(tend / 0.35) * 0.35)    # whole output intervals of 0.35
    return max(1, int(tend // T)) * T


def deck(mode, g, T, nu_zero, const):
    tmax = duration(mode, T, const)
    dtout = 0.35 if const else T / 8
    dt = min(0.1, (T if not const else 3.5) / 50)
    kc = KC[(mode, g)]
    s = '&mesh  nx = 1, ny = %d, nz = 1, alfa0 = %.17g, beta0 = %.17g, ly = 2.0 /\n' % (NY, 2 * np.pi / 15, KZ)
    s += '&physics  re = %g, S = 1.0, linear = .true., exact_shift = .true., s2_amplitude = 1.0, s2_period = %g /\n' % (
        1e12 if nu_zero else 1000.0, 0.0 if const else T)
    s += '&time_control  deltat = %.10g, cflmax = 0.0, t_max = %.10g, nstep = 100000, dt_stat = 1e9, dt_field = 1e9, dt_save = 1e9, time = 0.0, time_from_restart = .false. /\n' % (dt, tmax)
    s += '&init  amplitude = 0.05, seed = 1, kpeak = 6.0 /\n'
    if kc:
        s += '&production  prod_kc = %g, prod_edge = %g /\n' % (kc, 0.3 if g == 'h' else 0.0)
    s += "&kelvin  ix_mode = %d, iz_mode = 1, iy_mode = 1, dt_out = %.10g, refine = 2, tol = 1e9, csv = 'run.csv' /\n" % (
        0 if mode == 'a' else 1, dtout)
    return s, tmax, dt


LINE = re.compile(r'dt =\s+(\S+)\s+err v\s+(\S+)\s+eta\s+(\S+)\s+energy\s+(\S+)\s+proj\s+(\S+)\s+G=0 gain\s+(\S+)')


def run_case(build, mode, g, T, nu_zero, const):
    text, tmax, dt = deck(mode, g, T, nu_zero, const)
    with tempfile.TemporaryDirectory() as w:
        open(os.path.join(w, 'case.in'), 'w').write(text)
        r = subprocess.run(['mpirun', '-np', '1', build, 'case.in'], cwd=w, capture_output=True, text=True)
        rows = [[float(x) for x in m.groups()] for m in LINE.finditer(r.stdout)]
        gm = re.search(r'G =\s+(\S+),', r.stdout)
        gg = re.search(r'gamma_z\(closed form\)\| =\s+(\S+)', r.stdout)
        if len(rows) != 3:
            return dict(failed=True, log=r.stdout[-400:] + r.stderr[-400:])
        # the table of the finest run against kelvin_ref.py (an independent integration)
        par, d = K.read_csv(os.path.join(w, 'run.csv'))
        m = K.Mode(par['kx'][0], par['kz'][0], par['ky0'][0], par['G'][0], par['S'][0], par['s2_amplitude'][0],
                   par['s2_period'][0], par['nu'][0], complex(*par['v0']), complex(*par['eta0']))
        ref = m.series(d['t'])
        ev, ee, eE = K.errors(ref, d['Re_v'] + 1j * d['Im_v'], d['Re_eta'] + 1j * d['Im_eta'], d['energy'])
    a = np.array(rows)
    return dict(failed=False, G=float(gm.group(1)), tmax=tmax, dt=a[:, 0], v=a[:, 1], eta=a[:, 2], en=a[:, 3],
                proj=a[:, 4], gamma=float(gg.group(1)), py=max(ev, ee, eE))


def order(e, floor=5e-7):
    o = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1) if e[i] > floor and e[i + 1] > 0]
    return min(o) if o else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', default='build-cpu')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--quick', action='store_true', help='S T = 5.25 and nu = 1e-3 only')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    exe = os.path.join(os.path.dirname(here), a.build, 'test_kelvin_s2')
    cases = [(m, g, T, nz) for m in 'ab' for g in '0h1' for T in ((5.25,) if a.quick else (1.75, 5.25, 14.0))
             for nz in ((False,) if a.quick else (False, True))]
    controls = sorted({(m, g, 0.0, False) for m in 'ab' for g in '0h1'})
    jobs = [(m, g, T, nz, False) for m, g, T, nz in cases] + [(m, g, T, nz, True) for m, g, T, nz in controls]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: run_case(exe, *j), jobs))
    out = []
    ctrl = {(j[0], j[1]): r for j, r in zip(jobs, res) if j[4]}
    ok = True
    out.append('| mode | G | S T | nu | t_end | err v (3 dt) | err eta | err energy | proj | order v / eta / energy | control order | python | |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for j, r in zip(jobs, res):
        m, g, T, nz, const = j
        if r['failed']:
            out.append('| %s | %s | %g | %s | run failed: %s |' % (m, g, T, '0' if nz else '1e-3', r['log'].replace('\n', ' ')))
            ok = False
            continue
        ov, oe, on = order(r['v']), order(r['eta']), order(r['en'])
        c = ctrl[(m, g)]
        co = order(c['v']) if not c['failed'] else np.nan
        good = (max(r['v'][-1], r['eta'][-1], r['en'][-1]) <= 1e-5 and r['proj'].max() <= 5e-8 and r['gamma'] <= 1e-10
                and r['py'] <= 1e-5 and (np.isnan(ov) or ov >= 1.6) and (np.isnan(oe) or oe >= 1.6))
        if not const and not np.isnan(ov) and not np.isnan(co) and ov < co - 0.4:
            good = False
        ok = ok and good
        out.append('| %s%s | %.3f | %s | %s | %.1f | %s | %s | %s | %.0e | %.2f / %.2f / %.2f | %.2f | %.0e | %s |' % (
            m, ' (const S2)' if const else '', r['G'], '-' if const else '%g' % T, '0' if nz else '1e-3', r['tmax'],
            ' '.join('%.0e' % x for x in r['v']), ' '.join('%.0e' % x for x in r['eta']),
            ' '.join('%.0e' % x for x in r['en']), r['proj'].max(), ov, oe, on, co, r['py'], 'ok' if good else 'FAIL'))
    text = '\n'.join(out) + '\n\n' + ('ALL CASES PASS' if ok else 'SOME CASES FAIL') + '\n'
    print(text)
    if a.out:
        open(a.out, 'w').write(text)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
