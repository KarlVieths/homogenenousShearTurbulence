"""Readers and statistics of the robustness campaign (used by analyse.py, plain numpy).

A run directory holds hst.in and statistics/{Runtimedata, variances_runtime.dat, production_runtime.dat,
spectra/exz<i>.dat} (hst_stats.f90).  Conventions are those of the seed ensemble's analyse_seeds.py and of
doc.production/assets/campaign/tools/analyse_small_box.py: lab-frame second moments rotated into the shear
frame e_par = (cos th, sin th) in (x, z); a time that occurs twice in the appended files (a restarted run)
keeps its last record.  Everything is normalised with S_tot and lambda_c = 2 pi/prod_kc:

    q2n = q2/(S lambda_c)^2,  epsn = eps/(S^3 lambda_c^2),  S* = S q2/eps,  Leps = (q2/3)^(3/2)/eps/lambda_c,
    Re_lambda = (q2/3) sqrt(15/(nu eps)),  Re_c = S lambda_c^2/nu.
"""
import glob
import io
import math
import os
import re

import numpy as np

QUANT = ['q2n', 'epsn', 'Sstar', 'uv', 'Leps', 'Relam', 'b_par', 'b_v', 'b_perp', 'uperpv', 'rho_pp', 'E_low']
LABEL = {'q2n': 'q2/(S lc)^2', 'epsn': 'eps/(S^3 lc^2)', 'Sstar': 'S*', 'uv': '-<u_par v>/q2', 'Leps': 'L_eps/lc',
         'Relam': 'Re_lambda', 'b_par': 'b_par', 'b_v': 'b_vv', 'b_perp': 'b_perp', 'uperpv': '<u_perp v>/q2',
         'rho_pp': 'rho_par,perp', 'E_low': 'unforced q2 share'}
BLOCKS = [25.0, 50.0, 100.0, 200.0]


def loadtxt(path):
    # Fortran writes |x| < 1e-99 as 1.234-100 (no E): put the E back (slow, only when the plain read fails)
    try:
        return np.loadtxt(path, ndmin=2)
    except ValueError:
        pass
    with open(path) as f:
        text = re.sub(r'(\d)([+-]\d{3})(?=\s|$)', r'\1E\2', f.read())
    return np.loadtxt(io.StringIO(text), ndmin=2)


def deckval(path, key):
    m = re.search(r'^\s*' + key + r'\s*=\s*([-0-9.eEdD+]+)', open(path).read(), re.M)
    if m is None:
        raise KeyError(f'{key} not in {path}')
    return float(m.group(1).replace('d', 'e').replace('D', 'e'))


def chi2_sf(x, k):
    """P(chi^2_k > x): the regularised upper incomplete gamma function, series / continued fraction."""
    a, x2 = k/2.0, x/2.0
    if x2 <= 0:
        return 1.0
    if x2 < a + 1:
        term = s = 1.0/a
        for n in range(1, 500):
            term *= x2/(a + n); s += term
            if abs(term) < abs(s)*1e-14:
                break
        return 1.0 - s*math.exp(-x2 + a*math.log(x2) - math.lgamma(a))
    b, c, d = x2 + 1 - a, 1e300, 1.0/(x2 + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i*(i - a); b += 2
        d = an*d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an/c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0/d; delta = d*c; h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h*math.exp(-x2 + a*math.log(x2) - math.lgamma(a))


class Run:
    """The deck parameters and the time series of one run."""

    def __init__(self, d):
        self.dir = os.path.normpath(d)
        self.name = os.path.basename(self.dir)
        deck = os.path.join(d, 'hst.in')
        S, S2 = deckval(deck, 'S'), deckval(deck, 's2_amplitude')
        self.S0 = math.hypot(S, S2)
        self.S, self.S2 = S, S2
        self.theta = math.degrees(math.atan2(S2, S))
        self.ly, self.nu = deckval(deck, 'ly'), 1.0/deckval(deck, 're')
        self.lx, self.lz = 2*math.pi/deckval(deck, 'alfa0'), 2*math.pi/deckval(deck, 'beta0')
        self.edge = deckval(deck, 'prod_edge')
        self.lam = 2*math.pi/deckval(deck, 'prod_kc')
        self.Re_c = self.S0*self.lam**2/self.nu
        self.seed = int(deckval(deck, 'seed'))
        self.done = os.path.exists(os.path.join(d, 'DONE'))
        st = os.path.join(d, 'statistics')
        rt = loadtxt(os.path.join(st, 'Runtimedata'))
        var = loadtxt(os.path.join(st, 'variances_runtime.dat'))
        pf = loadtxt(os.path.join(st, 'production_runtime.dat'))
        n = min(len(rt), len(var), len(pf))
        rt, var, pf = rt[:n], var[:n], pf[:n]
        keep = np.zeros(n, bool); tmin = np.inf
        for i in range(n - 1, -1, -1):
            if rt[i, 0] < tmin - 1e-9:
                keep[i] = True; tmin = rt[i, 0]
        rt, var, pf = rt[keep], var[keep], pf[keep]
        c, s = math.cos(math.radians(self.theta)), math.sin(math.radians(self.theta))
        ly = self.ly
        uv, wv = 2*rt[:, 11]/ly, 2*rt[:, 12]/ly
        uu, ww, vv, uw = var[:, 1]/ly, var[:, 2]/ly, var[:, 3]/ly, var[:, 4]/ly
        self.t = rt[:, 0]
        self.vv = vv
        self.ser = dict(
            q2=2*rt[:, 9]/ly, eps=self.nu*2*rt[:, 10]/ly,
            upar_v=c*uv + s*wv, uperp_v=-s*uv + c*wv,
            upar2=c*c*uu + 2*c*s*uw + s*s*ww, uperp2=s*s*uu - 2*c*s*uw + c*c*ww, vv=vv,
            upar_uperp=c*s*(ww - uu) + (c*c - s*s)*uw,
            elow=(pf[:, 2] + pf[:, 3] + pf[:, 4])/rt[:, 9])

    def derived(self, m):
        q2, eps, S0, lam = m['q2'], m['eps'], self.S0, self.lam
        return dict(q2n=q2/(S0*lam)**2, epsn=eps/(S0**3*lam**2), Sstar=S0*q2/eps, uv=-m['upar_v']/q2,
                    Leps=(q2/3)**1.5/eps/lam, Relam=(q2/3)*math.sqrt(15/(self.nu*eps)),
                    b_par=m['upar2']/q2 - 1/3, b_v=m['vv']/q2 - 1/3, b_perp=m['uperp2']/q2 - 1/3,
                    uperpv=m['uperp_v']/q2, rho_pp=m['upar_uperp']/math.sqrt(m['upar2']*m['uperp2']),
                    E_low=m['elow'])

    def stats(self, t0, block):
        """(means, standard errors) for t >= t0; the errors from blocks of `block` time units (nan with < 3 blocks)."""
        t, keys = self.t, list(self.ser)
        sel = t >= t0
        if sel.sum() < 10:
            return None, None
        res = self.derived({k: self.ser[k][sel].mean() for k in keys})
        nb = int((t[-1] - t0)//block)
        err = {k: float('nan') for k in res}
        if nb >= 3:
            blocks = []
            for b in range(nb):
                m = (t >= t0 + b*block) & (t < t0 + (b + 1)*block)
                blocks.append(self.derived({k: self.ser[k][m].mean() for k in keys}))
            for k in res:
                err[k] = float(np.std([bl[k] for bl in blocks], ddof=1)/math.sqrt(nb))
        return res, err

    def q2_level(self, window=50.0):
        """q2 over the last `window` time units and over the window before (the laminar check)."""
        t = self.t
        a = self.ser['q2'][t > t[-1] - window].mean()
        m = (t > t[-1] - 2*window) & (t <= t[-1] - window)
        return a, (self.ser['q2'][m].mean() if m.any() else float('nan'))

    def tau(self, t0, h=0.5):
        """Integral autocorrelation time of q2(t) after t0 (Sokal window, 6 tau), in time units."""
        sel = self.t >= t0
        if sel.sum() < 10:
            return float('nan')
        g = np.arange(self.t[sel][0], self.t[-1], h)
        x = np.interp(g, self.t[sel], self.ser['q2'][sel]); x -= x.mean()
        n = len(x)
        f = np.fft.rfft(x, 2*n)
        ac = np.fft.irfft(f*np.conj(f))[:n]; ac /= ac[0]
        tau = 0.5
        for m in range(1, n):
            tau += ac[m]
            if m >= 6*tau:
                break
        return tau*h

    # ---- box-cycle lines in <v^2>(t)
    def box_periods(self):
        """The recurrence times of the periodic box: Lx/(S_x Ly) along x, Lz/(S_z Ly) along z (S_x = S, S_z = s2_amplitude)."""
        out = []
        if abs(self.S) > 1e-12:
            out.append(('x', self.lx/(abs(self.S)*self.ly)))
        if abs(self.S2) > 1e-12:
            out.append(('z', self.lz/(abs(self.S2)*self.ly)))
        return out

    def vv_spectrum(self, t0, seg=100.0, h=0.1):
        """Welch estimate of the power spectrum of <v^2>(t) after t0: (frequency [1/time], power, number of segments)."""
        sel = self.t >= t0
        if sel.sum() < 100 or self.t[-1] - t0 < 2*seg:
            return None
        g = np.arange(t0, self.t[-1], h)
        x = np.interp(g, self.t[sel], self.vv[sel]); x -= x.mean()
        ns = int(round(seg/h)); step = ns//2
        w = np.hanning(ns); w /= np.sqrt((w**2).sum())
        ps = []
        for i in range(0, len(x) - ns + 1, step):
            seg_x = x[i:i + ns]
            seg_x = seg_x - seg_x.mean()
            ps.append(np.abs(np.fft.rfft(seg_x*w))**2)
        return np.fft.rfftfreq(ns, h), np.mean(ps, axis=0), len(ps)


def line_ratio(freq, power, f0, width=1, back=0.1, gap=3):
    """Power in the bins within `width` of f0 over the mean of the neighbouring bins (f0 +- `back`, `gap` bins left out)."""
    df = freq[1] - freq[0]
    k0 = int(round(f0/df))
    if k0 + gap + 1 >= len(freq) or k0 - width < 1:
        return float('nan')
    line = power[k0 - width:k0 + width + 1].mean()
    nb = int(round(back/df))
    sel = [k for k in range(max(1, k0 - nb), min(len(freq), k0 + nb + 1)) if abs(k - k0) > gap]
    return float(line/power[sel].mean()) if sel else float('nan')


def read_spectra(d, t_min):
    """Average of the column spectra exz<i>.dat of a run whose averaging interval ends after t_min:
    (kx, kz, E, number of files).  The files hold E(ix, iz) for ix = 0..nx, iz = -nz..nz with the ix > 0 modes counted twice."""
    acc, n, kx, kz = None, 0, None, None
    for f in sorted(glob.glob(os.path.join(d, 'statistics', 'spectra', 'exz*.dat'))):
        with open(f) as fh:
            head = fh.readline()
        m = re.search(r'over t =\s*([-0-9.eE+]+)\s*\.\.\s*([-0-9.eE+]+)', head)
        if not m or float(m.group(1)) < t_min - 1e-9:
            continue
        a = np.loadtxt(f, ndmin=2)
        if acc is None:
            acc, kx, kz = np.zeros(len(a)), a[:, 2], a[:, 3]
        acc += a[:, 4]; n += 1
    return (kx, kz, acc/n, n) if n else None


def correlation(kx, kz, E, dx, dz):
    """R(dx, dz) = sum E cos(kx dx + kz dz) / sum E (the correlation of the fluctuation, u_i u_i summed)."""
    return float((E*np.cos(kx*dx + kz*dz)).sum()/E.sum())
