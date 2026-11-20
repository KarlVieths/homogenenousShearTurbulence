#!/usr/bin/env python3
"""Seed ensemble: per-run statistics with their error bars, checks of the error bars, and the
scatter of the run means over the seeds.

    python3 analyse_seeds.py [--runs runs] [--transient 400] [--block 50] [--extra DIR ...]
                             [--lf 1.7320508]                           (from the campaign directory)

Per run (runs/<name>/, files in statistics/):
  1. time means over t >= --transient and the standard error from blocks of --block
     time units, for q2, eps, S* = S q2/eps, -<u_par v>/q2, L_eps/l_f, b_par, b_vv, b_perp,
     <u_perp v>/q2, rho_par,perp and the share of q2 in the unforced range (columns with
     G < 1, production_runtime.dat);
  2. checks of the error bars: the standard error for block lengths 25, 50, 100, 200 (it
     should plateau), the integral autocorrelation time tau of q2(t), and the means with the
     window starting at --transient2 (600) instead.
Across the runs, per quantity, with run means x_i and standard errors SE_i (N runs):
  T = sum ((x_i - mean)/SE_i)^2 ~ chi^2(N-1), p = P(chi^2 > T), R = sqrt(T/(N-1)): the factor by
  which to inflate the quoted errors; the seed-to-seed standard deviation, and 2.77 sd, the
  difference of two single runs that is significant at about 95 %.
--extra DIR adds a finished or running run of the same configuration (e.g. the in-progress
square-box run at 30 deg) as a further sample.
Writes seed_runs.csv (one row per run), seed_blocks.csv (SE against the block length) and
seed_ensemble.csv (one row per quantity), and prints them.

Conventions as in doc.production/assets/campaign/tools/analyse_small_box.py (hst_stats.f90):
lab-frame second moments rotated into the shear frame e_par = (cos th, sin th) in (x, z);
nu = 1/Re, ly, S, s2_amplitude from the run's deck; the files of a restarted run are
appended, so a time that occurs twice keeps its last record.
"""
import argparse, glob, io, math, os, re
import numpy as np

QUANT = ['q2', 'eps', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v', 'b_perp', 'uperpv', 'rho_pp', 'E_low']
LABEL = {'q2': 'q2', 'eps': 'eps', 'Sstar': 'S*', 'uv': '-<u_par v>/q2', 'Leps': 'L_eps/l_f', 'b_par': 'b_par',
         'b_v': 'b_vv', 'b_perp': 'b_perp', 'uperpv': '<u_perp v>/q2', 'rho_pp': 'rho_par,perp',
         'E_low': 'unforced q2 share'}
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
    return float(m.group(1).replace('d', 'e').replace('D', 'e'))


def load(d):
    st = os.path.join(d, 'statistics')
    rt = loadtxt(os.path.join(st, 'Runtimedata'))
    var = loadtxt(os.path.join(st, 'variances_runtime.dat'))
    pf = loadtxt(os.path.join(st, 'production_runtime.dat'))
    n = min(len(rt), len(var), len(pf))
    rt, var, pf = rt[:n], var[:n], pf[:n]
    keep = np.zeros(n, bool); tmin = np.inf
    for i in range(n - 1, -1, -1):                     # keep the last record of every time
        if rt[i, 0] < tmin - 1e-9:
            keep[i] = True; tmin = rt[i, 0]
    return rt[keep], var[keep], pf[keep]


class Run:
    def __init__(self, d):
        self.dir, self.name = d, os.path.basename(os.path.normpath(d))
        deck = os.path.join(d, 'hst.in')
        S, S2 = deckval(deck, 'S'), deckval(deck, 's2_amplitude')
        self.ly, self.nu = deckval(deck, 'ly'), 1.0/deckval(deck, 're')
        self.S0 = math.hypot(S, S2)
        self.theta = math.degrees(math.atan2(S2, S))
        self.seed = int(deckval(deck, 'seed'))
        self.done = os.path.exists(os.path.join(d, 'DONE'))
        rt, var, pf = load(d)
        c, s = math.cos(math.radians(self.theta)), math.sin(math.radians(self.theta))
        ly = self.ly
        uv, wv = 2*rt[:, 11]/ly, 2*rt[:, 12]/ly
        uu, ww, vv, uw = var[:, 1]/ly, var[:, 2]/ly, var[:, 3]/ly, var[:, 4]/ly
        self.t = rt[:, 0]
        self.ser = dict(
            q2=2*rt[:, 9]/ly, eps=self.nu*2*rt[:, 10]/ly,
            upar_v=c*uv + s*wv, uperp_v=-s*uv + c*wv,
            upar2=c*c*uu + 2*c*s*uw + s*s*ww, uperp2=s*s*uu - 2*c*s*uw + c*c*ww, vv=vv,
            upar_uperp=c*s*(ww - uu) + (c*c - s*s)*uw,
            elow=(pf[:, 2] + pf[:, 3] + pf[:, 4])/rt[:, 9])      # (1-G) <u_i u_i> / <u_i u_i>, both ly/2 <.>

    def derived(self, m, lf):
        q2, eps = m['q2'], m['eps']
        return dict(q2=q2, eps=eps, Sstar=self.S0*q2/eps, uv=-m['upar_v']/q2, Leps=(q2/3)**1.5/eps/lf,
                    b_par=m['upar2']/q2 - 1/3, b_v=m['vv']/q2 - 1/3, b_perp=m['uperp2']/q2 - 1/3,
                    uperpv=m['uperp_v']/q2, rho_pp=m['upar_uperp']/math.sqrt(m['upar2']*m['uperp2']),
                    E_low=m['elow'])

    def stats(self, t0, block, lf):
        """(means, standard errors) over t >= t0, errors from blocks of `block` time units."""
        t, keys = self.t, list(self.ser)
        sel = t >= t0
        if sel.sum() < 10:
            return None, None
        res = self.derived({k: self.ser[k][sel].mean() for k in keys}, lf)
        nb = int((t[-1] - t0)//block)
        err = {k: float('nan') for k in res}
        if nb >= 3:
            blocks = []
            for b in range(nb):
                m = (t >= t0 + b*block) & (t < t0 + (b + 1)*block)
                blocks.append(self.derived({k: self.ser[k][m].mean() for k in keys}, lf))
            for k in res:
                err[k] = np.std([bl[k] for bl in blocks], ddof=1)/math.sqrt(nb)
        return res, err

    def tau(self, t0, h=0.5):
        """Integral autocorrelation time of q2(t) after t0, windowed at 6 tau (Sokal), time units."""
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


def chi2_sf(x, k):
    """P(chi^2_k > x): the regularised upper incomplete gamma function, series / continued fraction."""
    a, x2 = k/2.0, x/2.0
    if x2 <= 0:
        return 1.0
    if x2 < a + 1:                                   # series for the lower function
        term = s = 1.0/a
        for n in range(1, 500):
            term *= x2/(a + n); s += term
            if abs(term) < abs(s)*1e-14:
                break
        return 1.0 - s*math.exp(-x2 + a*math.log(x2) - math.lgamma(a))
    b, c, d = x2 + 1 - a, 1e300, 1.0/(x2 + 1 - a)    # Lentz continued fraction for the upper function
    h = d
    for i in range(1, 500):
        an = -i*(i - a); b += 2
        d = an*d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an/c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0/d; delta = d*c; h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h*math.exp(-x2 + a*math.log(x2) - math.lgamma(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--transient', type=float, default=400.0)
    ap.add_argument('--transient2', type=float, default=600.0, help='second window start, for the check')
    ap.add_argument('--block', type=float, default=50.0)
    ap.add_argument('--lf', type=float, default=math.sqrt(3.0), help='forcing wavelength 2 pi/k_c')
    ap.add_argument('--extra', action='append', default=[], help='another run directory of the same configuration')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = sorted(glob.glob(os.path.join(here, a.runs, '*', '')))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, 'statistics', 'Runtimedata'))] + a.extra
    runs = []
    for d in dirs:
        try:
            runs.append(Run(d))
        except (OSError, ValueError, IndexError) as e:
            print(f'{d}: not read ({e})')
    out = []
    for r in runs:
        res, err = r.stats(a.transient, a.block, a.lf)
        if res is None:
            print(f'{r.name}: t = {r.t[-1]:.0f}, shorter than the transient {a.transient:g}: skipped')
            continue
        res2, _ = r.stats(a.transient2, a.block, a.lf)
        out.append((r, res, err, res2))
    if not out:
        print('no run has passed the transient yet'); return

    # per run: means, errors, second window, tau
    with open(os.path.join(here, 'seed_runs.csv'), 'w') as f:
        f.write('run,seed,theta,t_end,done,tau_q2,' + ','.join(QUANT) + ',' + ','.join('se_' + k for k in QUANT)
                + ',' + ','.join(a_ + 'w2' for a_ in QUANT) + '\n')
        for r, res, err, res2 in out:
            w2 = [res2[k] if res2 else float('nan') for k in QUANT]
            f.write(f'{r.name},{r.seed},{r.theta:.2f},{r.t[-1]:.1f},{int(r.done)},{r.tau(a.transient):.3g},'
                    + ','.join(f'{res[k]:.6g}' for k in QUANT) + ',' + ','.join(f'{err[k]:.3g}' for k in QUANT)
                    + ',' + ','.join(f'{x:.6g}' for x in w2) + '\n')
    print(f'{len(out)} runs, averages after t = {a.transient:g}, blocks of {a.block:g}')
    for r, res, err, res2 in out:
        print(f'  {r.name:22s} seed {r.seed:4d}  t = {r.t[-1]:6.0f}{"" if r.done else " (running)"}  tau(q2) = {r.tau(a.transient):5.1f}'
              f'  q2 {res["q2"]:.4f}({err["q2"]:.4f})  S* {res["Sstar"]:.3f}({err["Sstar"]:.3f})')

    # the check of the error bars: SE against the block length
    print('\nstandard error against the block length (it should plateau), and the mean from t >= '
          f'{a.transient2:g}:')
    with open(os.path.join(here, 'seed_blocks.csv'), 'w') as f:
        f.write('run,quantity,' + ','.join(f'se_block{b:g}' for b in BLOCKS) + f',mean_from{a.transient:g},mean_from{a.transient2:g}\n')
        for r, res, err, res2 in out:
            se = [r.stats(a.transient, b, a.lf)[1] for b in BLOCKS]
            for k in QUANT:
                f.write(f'{r.name},{k},' + ','.join(f'{e[k]:.3g}' for e in se) + f',{res[k]:.6g},{res2[k] if res2 else float("nan"):.6g}\n')
            print(f'  {r.name} ({r.t[-1]:.0f})')
            print(f'    {"":16s}' + ''.join(f'{"SE@" + format(b, "g"):>10s}' for b in BLOCKS) + f'{"mean":>11s}{"mean@" + format(a.transient2, "g"):>11s}')
            for k in ('q2', 'Sstar', 'uv', 'Leps'):
                print(f'    {LABEL[k]:16s}' + ''.join(f'{e[k]:10.3g}' for e in se) + f'{res[k]:11.4f}{(res2[k] if res2 else float("nan")):11.4f}')

    # across the runs
    n = len(out)
    if n < 2:
        print('\nneed at least two runs for the ensemble statistics'); return
    print(f'\nacross the {n} runs: seed-to-seed scatter against the quoted errors')
    print(f'{"quantity":20s}{"mean":>10s}{"sd":>10s}{"sd/mean":>9s}{"<SE>":>10s}{"T":>8s}{"p":>8s}{"R":>7s}{"2.77 sd":>10s}')
    with open(os.path.join(here, 'seed_ensemble.csv'), 'w') as f:
        f.write('quantity,N,mean,sd,mean_SE,T,p,R,diff95\n')
        for k in QUANT:
            x = np.array([o[1][k] for o in out]); se = np.array([o[2][k] for o in out])
            ok = np.isfinite(se) & (se > 0)
            T = float(np.sum(((x - x.mean())/se)**2)) if ok.all() else float('nan')
            sd = x.std(ddof=1)
            p = chi2_sf(T, n - 1) if np.isfinite(T) else float('nan')
            R = math.sqrt(T/(n - 1)) if np.isfinite(T) else float('nan')
            f.write(f'{k},{n},{x.mean():.6g},{sd:.4g},{np.nanmean(se):.3g},{T:.4g},{p:.3g},{R:.4g},{2.77*sd:.4g}\n')
            print(f'{LABEL[k]:20s}{x.mean():10.4f}{sd:10.4f}{sd/abs(x.mean()):9.3f}{np.nanmean(se):10.4f}{T:8.2f}{p:8.3f}{R:7.2f}{2.77*sd:10.4f}')
    lo, hi = 0.0, 100.0                              # the 5 % point of chi^2(N-1) by bisection
    for _ in range(80):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if chi2_sf(mid, n - 1) > 0.05 else (lo, mid)
    print(f'\nT ~ chi^2({n - 1}); T above {hi:.1f} is significant at 5 %')
    print('R = sqrt(T/(N-1)) multiplies the quoted errors; with N = 4 the sd is uncertain by about 40 %,')
    print('so R is a rough calibration: say whether it varies between the quantities.')
    print('seed_runs.csv, seed_blocks.csv and seed_ensemble.csv written')


if __name__ == '__main__':
    main()
