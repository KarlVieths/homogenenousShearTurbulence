#!/usr/bin/env python3
"""Reference solution of a Kelvin mode in an oscillating spanwise shear, with
the production filter weight G of its column (tests/KELVIN_S2.md).

U = (S y, 0, S2(t) y),  S2(t) = s2_mean + A sin(2 pi (t - s2_start)/T)  (t >= s2_start)
mode  u(x, t) = uhat(t) exp(i (kx x + ky(t) y + kz z)):

  ky(t)  = ky0 - kx S t - kz gamma_z(t),         gamma_z = int S2 dt
  vhat   = v0 (k0^2/k^2)^((1+G)/2) E(t),         E = exp(-nu int k^2 dt)
  etahat = E [eta0 - i G v0 int (S kz - S2 kx) (k0^2/k^2)^((1+G)/2) dt]
  energy = 1/2 |uhat|^2 = 1/2 (k^2 |vhat|^2 + |etahat|^2)/kh^2

    kelvin_ref.py --selftest
        closed forms against a direct integration of the three velocity
        components with the pressure from continuity
    kelvin_ref.py --kx 0 --kz 1.2566370614 --ky0 3.1415926536 --G 0 --S 1 --A 1 \
        --T 5.25 --nu 1e-3 --v0 1+0.3j --eta0 0.4-0.2j --periods 10 --out ref.csv
        the reference time series (t, S2, gamma_z, ky, E, Re_v, Im_v, Re_eta, Im_eta, energy)
    kelvin_ref.py --compare run.csv [--tol 1e-6]
        the table written by tests/test_kelvin_s2 (csv = 'run.csv', whose header
        holds the parameters) against this reference; exit status 1 above tol

Errors in v, eta and the energy are relative to the largest reference
amplitude of the run, as in the Fortran test.  Only numpy and scipy.
"""
import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp

TWOPI = 2.0 * np.pi


class Mode:
    """One column: parameters and the closed forms."""

    def __init__(self, kx, kz, ky0, G, S, A, T, nu, v0, eta0, mean=0.0, start=0.0, t0=0.0):
        self.kx, self.kz, self.ky0, self.G, self.S = kx, kz, ky0, G, S
        self.A, self.T, self.nu, self.mean, self.start, self.t0 = A, T, nu, mean, start, t0
        self.v0, self.eta0 = complex(v0), complex(eta0)
        self.kh2 = kx * kx + kz * kz
        self.k02 = self.kh2 + self.ky(t0) ** 2     # ky0 is the wavenumber at t = 0, k0 that at t0
        self.p = 0.5 * (1.0 + G)

    def s2(self, t):
        t = np.asarray(t, dtype=float)
        if self.T > 0:
            s = self.mean + self.A * np.sin(TWOPI * (t - self.start) / self.T)
        else:
            s = (self.mean + self.A) * np.ones_like(t)
        return np.where(t >= self.start, s, 0.0)

    def gamma(self, t):
        t = np.asarray(t, dtype=float)
        u = np.maximum(t - self.start, 0.0)
        if self.T > 0:
            om = TWOPI / self.T
            return self.mean * u + self.A / om * (1.0 - np.cos(om * u))
        return (self.mean + self.A) * u

    def ky(self, t):
        return self.ky0 - self.kx * self.S * np.asarray(t, dtype=float) - self.kz * self.gamma(t)

    def k2(self, t):
        return self.kh2 + self.ky(t) ** 2

    def integrals(self, t):
        """int_t0^t k^2 and int_t0^t (S kz - S2 kx)(k0^2/k^2)^p, for sorted t >= t0."""
        def rhs(tau, y):
            k2 = self.k2(tau)
            return [k2, (self.S * self.kz - self.s2(tau) * self.kx) * (self.k02 / k2) ** self.p]
        t = np.asarray(t, dtype=float)
        out = np.zeros((2, t.size))
        if t.size > 1:
            sol = solve_ivp(rhs, (t[0], t[-1]), [0.0, 0.0], t_eval=t, method='DOP853', rtol=1e-13, atol=1e-15)
            out = sol.y
        return out

    def series(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self.integrals(t)
        k2 = self.k2(t)
        E = np.exp(-self.nu * a)
        v = self.v0 * (self.k02 / k2) ** self.p * E
        eta = E * (self.eta0 - 1j * self.G * self.v0 * b)
        en = 0.5 * (k2 * np.abs(v) ** 2 + np.abs(eta) ** 2) / self.kh2
        return dict(t=t, S2=self.s2(t), gamma=self.gamma(t), ky=self.ky(t), E=E, v=v, eta=eta, energy=en, a=a)

    def direct(self, t):
        """The three velocity components integrated directly, with the pressure from continuity:
        uhat' = -G v (S, 0, S2) + (1 + G) a v k/k^2 - nu k^2 uhat,  a = S kx + S2 kz."""
        kx, kz, G, S, nu = self.kx, self.kz, self.G, self.S, self.nu
        ky0 = self.ky(self.t0)
        u0 = (-kx * ky0 * self.v0 - 1j * kz * self.eta0) / self.kh2
        w0 = (-kz * ky0 * self.v0 + 1j * kx * self.eta0) / self.kh2

        def rhs(tau, y):
            u, v, w = y[0] + 1j * y[3], y[1] + 1j * y[4], y[2] + 1j * y[5]
            ky = float(self.ky(tau))
            k2 = self.kh2 + ky * ky
            s2 = float(self.s2(tau))
            a = S * kx + s2 * kz
            f = (1 + G) * a * v / k2
            du = -G * v * S + f * kx - nu * k2 * u
            dv = f * ky - nu * k2 * v
            dw = -G * v * s2 + f * kz - nu * k2 * w
            return [du.real, dv.real, dw.real, du.imag, dv.imag, dw.imag]

        t = np.asarray(t, dtype=float)
        sol = solve_ivp(rhs, (t[0], t[-1]), [u0.real, self.v0.real, w0.real, u0.imag, self.v0.imag, w0.imag],
                        t_eval=t, method='DOP853', rtol=1e-13, atol=1e-16)
        u, v, w = sol.y[0] + 1j * sol.y[3], sol.y[1] + 1j * sol.y[4], sol.y[2] + 1j * sol.y[5]
        return u, v, w


def errors(ref, v, eta, energy):
    """Errors in v, eta, energy relative to the largest reference amplitude."""
    return (np.max(np.abs(v - ref['v'])) / np.max(np.abs(ref['v'])),
            np.max(np.abs(eta - ref['eta'])) / np.max(np.abs(ref['eta'])),
            np.max(np.abs(energy - ref['energy'])) / np.max(ref['energy']))


def selftest(tol=2e-8):
    kz = TWOPI / 5.0
    cases = []
    for kx in (0.0, TWOPI / 15.0):
        for G in (0.0, 0.547, 1.0):
            for ST in (1.75, 5.25, 14.0):
                cases.append((kx, G, ST))
    worst = 0.0
    print('closed forms against the direct integration of (u, v, w); relative to the peak amplitude')
    print('   kx      G     ST   periods   v         eta       energy    continuity  int k^2 (kx = 0)')
    for kx, G, ST in cases:
        periods = 10 if kx == 0.0 else 3
        m = Mode(kx, kz, np.pi, G, 1.0, 1.0, ST, 1e-3, 1 + 0.3j, 0.4 - 0.2j)
        t = np.linspace(0.0, periods * ST, 401)
        ref = m.series(t)
        u, v, w = m.direct(t)
        eta = 1j * kz * u - 1j * kx * w
        en = 0.5 * (np.abs(u) ** 2 + np.abs(v) ** 2 + np.abs(w) ** 2)
        ev, ee, eE = errors(ref, v, eta, en)
        cont = np.max(np.abs(kx * u + m.ky(t) * v + kz * w)) / np.max(np.abs(v))
        extra = ''
        if kx == 0.0:       # the closed form of the k^2 integral (section 3.1 of the specification)
            om = TWOPI / ST
            c = kz * 1.0 / om
            b = np.pi - c
            ik2 = (kz ** 2 + b ** 2) * t + 2 * c * b / om * np.sin(om * t) + c ** 2 * (t / 2 + np.sin(2 * om * t) / (4 * om))
            ea = np.max(np.abs(ik2 - ref['a'])) / np.max(ref['a'])
            extra = '%9.1e' % ea
            worst = max(worst, ea)
        print('%6.3f  %5.3f  %5.2f  %5d   %8.1e  %8.1e  %8.1e  %8.1e    %s' % (kx, G, ST, periods, ev, ee, eE, cont, extra))
        worst = max(worst, ev, ee, eE, cont)
        # a column with G = 0 gains no energy: |u|^2 = |u0|^2 E^2
        if G == 0.0:
            u0sq = np.abs(u[0]) ** 2 + np.abs(v[0]) ** 2 + np.abs(w[0]) ** 2
            gain = np.max(np.abs(2 * en - u0sq * ref['E'] ** 2)) / u0sq
            worst = max(worst, gain)
            if gain > tol:
                print('   G = 0 energy gain %.1e' % gain)
    print('worst: %.1e (limit %.0e)' % (worst, tol))
    return worst <= tol


def read_csv(path):
    par, rows, names = {}, [], None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                s = line[1:].split('=', 1)
                if len(s) == 2:
                    par[s[0].strip()] = [float(x) for x in s[1].split()]
            elif names is None:
                names = line.strip().split(',')
            elif line.strip():
                rows.append([float(x) for x in line.split(',')])
    tab = np.array(rows)
    return par, {n: tab[:, i] for i, n in enumerate(names)}


def compare(path, tol):
    par, d = read_csv(path)
    g = lambda k: par[k][0]
    m = Mode(g('kx'), g('kz'), g('ky0'), g('G'), g('S'), g('s2_amplitude'), g('s2_period'), g('nu'),
             complex(*par['v0']), complex(*par['eta0']), mean=g('s2_mean'), start=g('s2_start'), t0=g('t0'))
    # ky0 in the table is the wavenumber at t0; Mode wants it at t = 0
    m.ky0 = g('ky0') + m.kx * m.S * m.t0 + m.kz * float(m.gamma(m.t0))
    m.k02 = m.kh2 + float(m.ky(m.t0)) ** 2
    ref = m.series(d['t'])
    v = d['Re_v'] + 1j * d['Im_v']
    eta = d['Re_eta'] + 1j * d['Im_eta']
    ev, ee, eE = errors(ref, v, eta, d['energy'])
    eky = np.max(np.abs(d['ky'] - ref['ky']))
    eg = np.max(np.abs(d['gamma_code'] - ref['gamma']))
    es2 = np.max(np.abs(d['S2'] - ref['S2']))
    print('%s: %d times, G = %.4f   error v %.2e  eta %.2e  energy %.2e   ky %.1e  gamma %.1e  S2 %.1e'
          % (path, d['t'].size, m.G, ev, ee, eE, eky, eg, es2))
    return max(ev, ee, eE) <= tol and max(eky, eg, es2) <= 1e-10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--compare', metavar='CSV')
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--kx', type=float, default=0.0)
    ap.add_argument('--kz', type=float, default=TWOPI / 5.0)
    ap.add_argument('--ky0', type=float, default=np.pi)
    ap.add_argument('--G', type=float, default=0.0)
    ap.add_argument('--S', type=float, default=1.0)
    ap.add_argument('--A', type=float, default=1.0, help='amplitude of S2')
    ap.add_argument('--T', type=float, default=5.25, help='period of S2 (0: constant S2 = mean + A)')
    ap.add_argument('--mean', type=float, default=0.0, help='mean of S2')
    ap.add_argument('--start', type=float, default=0.0, help='S2 is zero before this time')
    ap.add_argument('--nu', type=float, default=1e-3)
    ap.add_argument('--v0', type=complex, default=1 + 0.3j)
    ap.add_argument('--eta0', type=complex, default=0.4 - 0.2j)
    ap.add_argument('--periods', type=float, default=10.0)
    ap.add_argument('--npts', type=int, default=2001)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.compare:
        sys.exit(0 if compare(a.compare, a.tol) else 1)
    m = Mode(a.kx, a.kz, a.ky0, a.G, a.S, a.A, a.T, a.nu, a.v0, a.eta0, mean=a.mean, start=a.start)
    t = np.linspace(0.0, a.periods * a.T if a.T > 0 else a.periods, a.npts)
    r = m.series(t)
    tab = np.column_stack([r['t'], r['S2'], r['gamma'], r['ky'], r['E'], r['v'].real, r['v'].imag,
                           r['eta'].real, r['eta'].imag, r['energy']])
    head = 't,S2,gamma_z,ky,E,Re_v,Im_v,Re_eta,Im_eta,energy'
    if a.out:
        np.savetxt(a.out, tab, delimiter=',', header=head, comments='', fmt='%.15e')
    else:
        np.savetxt(sys.stdout, tab, delimiter=',', header=head, comments='', fmt='%.15e')


if __name__ == '__main__':
    main()
