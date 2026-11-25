#!/usr/bin/env python3
"""Per-run statistics and the comparisons of the robustness campaign (planning document, "Outputs and analysis").

    python3 analyse.py --baseline <square-box run at 0 deg> [<at 15, 30, 45 deg> ...] [--transient 400] [--block 50]
                       [--sigma ../square_seed_ensemble/seed_ensemble.csv] [--inflate]

Needs numpy only; reads runs/<id>/ (statistics/ files) of every run that has them, and the baseline.

Baseline (the square box 10.5 x 2 sqrt 3 x 10.5, e = 0.3, Re_c = 3000): the runs given with --baseline (the in-progress
square-box campaign, 0/15/30/45 deg) and the stage-0 runs of ../square_seed_ensemble/runs (30 deg), found by their
parameters; at one angle the baseline is the mean of its runs.  sigma_seed is the seed-to-seed standard deviation of
stage 0 (--sigma, the `sd` column of seed_ensemble.csv, rescaled to the normalised quantities); without it a
difference is judged by the quoted errors (and --inflate multiplies those by R of stage 0, at least 1).

Per run (results.csv, blocks.csv): time means for t >= --transient with block errors, the blocking check (the error for blocks of
25, 50, 100, 200), the integral autocorrelation time of q2(t), the means from t >= 600; quantities normalised with S and
lambda_c: q2/(S lc)^2, eps/(S^3 lc^2), S*, -<u_par v>/q2, L_eps/lc, Re_lambda, b_par, b_vv, b_perp, <u_perp v>/q2,
rho_par,perp, the unforced share of q2.

Comparisons (each printed and written as a csv):
  box.csv          1a-c, 1e: difference to the baseline at the same angle, in units of sigma_seed; ok within 2.8 sigma_seed
  orientation.csv  all: x(30 deg) - x(0 deg) of a family (same id apart from the angle), ok within 2.8 sigma_seed; the cross terms
                   <u_perp v>/q2 and rho_par,perp at 30 deg against zero
  resonance.csv    all: lines of the <v^2>(t) spectrum at the box recurrence times Lx/(S Ly) (and Lz/(S2 Ly) with S2 != 0) and
                   their harmonics: power in the line bins over the neighbouring bins; flagged above --line-ratio
  decorrelation.csv  all: R(dx, 0) and R(0, dz) of the column spectra at half the box (and the largest |R| from a quarter box to half)
  edge.csv         stage 2: the quantities against e at A_h = 6.06 and 4.5 (e = 0.3 from the baseline / stage 1a) and the interaction
  reynolds.csv     stages 3a, 3b: Re_lambda, S*, L_eps/lc, q2/(S lc)^2 against Re_c, and whether the flow stays turbulent
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np

from hstlib import BLOCKS, LABEL, QUANT, Run, correlation, line_ratio, read_spectra

HERE = os.path.dirname(os.path.abspath(__file__))
BOXQ = ['q2n', 'epsn', 'Sstar', 'Leps', 'b_par', 'b_v', 'b_perp', 'E_low']
EDGEQ = ['q2n', 'Sstar', 'Leps', 'b_par', 'b_v', 'E_low']
REYQ = ['Relam', 'Sstar', 'Leps', 'q2n']
# stage-0 quantity names (analyse_seeds.py) -> ours, with the factor that normalises sd (S = 1, lambda_c = sqrt 3 there)
SIGMA_MAP = {'q2': 'q2n', 'eps': 'epsn', 'Sstar': 'Sstar', 'uv': 'uv', 'Leps': 'Leps', 'b_par': 'b_par', 'b_v': 'b_v',
             'b_perp': 'b_perp', 'uperpv': 'uperpv', 'rho_pp': 'rho_pp', 'E_low': 'E_low'}


class Entry:
    def __init__(self, run, stage, a):
        self.run, self.stage = run, stage
        self.res, self.err = run.stats(a.transient, a.block)
        self.res2, _ = run.stats(a.transient2, a.block)
        self.tau = run.tau(a.transient)
        self.name = run.name


def sigma_table(path, lam=math.sqrt(3.0), S0=1.0):
    """sigma_seed and R per (our) quantity from stage 0's seed_ensemble.csv."""
    sig, fac = {}, {}
    if not path or not os.path.exists(path):
        return sig, fac
    for r in csv.DictReader(open(path)):
        k = SIGMA_MAP.get(r['quantity'])
        if k is None:
            continue
        scale = {'q2n': 1/(S0*lam)**2, 'epsn': 1/(S0**3*lam**2)}.get(k, 1.0)
        # a guard for a degenerate stage 0 (sd far below the quoted error): sigma_seed is at least half the mean quoted error
        sig[k] = max(float(r['sd']), 0.5*float(r['mean_SE']))*scale
        fac[k] = max(1.0, float(r['R'])) if r['R'] not in ('', 'nan') else 1.0
    return sig, fac


def is_baseline_box(r):
    return (abs(r.lx - 10.5) < 1e-6 and abs(r.lz - 10.5) < 1e-6 and abs(r.ly - 2*math.sqrt(3)) < 1e-6
            and abs(r.edge - 0.3) < 1e-9 and abs(r.Re_c - 3000) < 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--baseline', nargs='*', default=[], help='run directories of the baseline box (the in-progress campaign)')
    ap.add_argument('--stage0', default=os.path.join(HERE, '..', 'square_seed_ensemble', 'runs'),
                    help='the stage-0 runs (baseline at 30 deg too); "" to leave them out')
    ap.add_argument('--sigma', default=os.path.join(HERE, '..', 'square_seed_ensemble', 'seed_ensemble.csv'))
    ap.add_argument('--inflate', action='store_true', help='multiply the quoted errors by R of stage 0 (at least 1)')
    ap.add_argument('--transient', type=float, default=400.0)
    ap.add_argument('--transient2', type=float, default=600.0)
    ap.add_argument('--block', type=float, default=50.0)
    ap.add_argument('--line-ratio', type=float, default=3.0, help='box-cycle line flagged above this power ratio')
    ap.add_argument('--decor-tol', type=float, default=0.1, help='|R| at half the box above this is flagged')
    ap.add_argument('--manifest', default='manifest.csv')
    ap.add_argument('--out', default='analysis', help='directory of the csv files')
    ap.add_argument('--stages', default='', help='only these stages (comparisons and results)')
    a = ap.parse_args()
    out = os.path.join(HERE, a.out); os.makedirs(out, exist_ok=True)
    man = {r['id']: r for r in csv.DictReader(open(os.path.join(HERE, a.manifest)))}
    sig, rfac = sigma_table(a.sigma)
    if not sig:
        print('note: no seed_ensemble.csv (stage 0): differences are judged by the quoted errors, not sigma_seed')

    # --- the runs
    def have(d):
        return os.path.exists(os.path.join(d, 'statistics', 'Runtimedata')) and os.path.exists(os.path.join(d, 'hst.in'))

    entries, base_all = [], []
    for d in sorted(glob.glob(os.path.join(HERE, a.runs, '*'))):
        rid = os.path.basename(d)
        if rid not in man or not have(d):
            continue
        if a.stages and man[rid]['stage'] not in a.stages.split(','):
            continue
        try:
            e = Entry(Run(d), man[rid]['stage'], a)
        except (OSError, ValueError, KeyError, IndexError) as ex:
            print(f'{rid}: not read ({ex})'); continue
        if e.res is None:
            print(f'{rid}: t = {e.run.t[-1]:.0f}, shorter than the transient {a.transient:g}: skipped'); continue
        entries.append(e)
    base_dirs = list(a.baseline) + ([d for d in glob.glob(os.path.join(a.stage0, '*'))] if a.stage0 else [])
    for d in base_dirs:
        if not have(d):
            continue
        try:
            e = Entry(Run(d), '0', a)
        except (OSError, ValueError, KeyError, IndexError) as ex:
            print(f'baseline {d}: not read ({ex})'); continue
        if e.res is not None and is_baseline_box(e.run):
            base_all.append(e)
        elif e.res is not None:
            print(f'baseline {d}: not the baseline box/filter/Re_c, ignored')
    if not entries:
        print('no run of the campaign has passed the transient yet'); return
    print(f'{len(entries)} runs, {len(base_all)} baseline runs (angles: '
          f'{sorted({round(b.run.theta, 1) for b in base_all})}), averages after t = {a.transient:g}, blocks of {a.block:g}\n')

    def infl(q):
        return rfac.get(q, 1.0) if a.inflate else 1.0

    def base_at(theta):
        bs = [b for b in base_all if abs(b.run.theta - theta) < 0.01]
        if not bs:
            return None
        res = {k: float(np.mean([b.res[k] for b in bs])) for k in QUANT}
        err = {k: math.sqrt(sum(b.err[k]**2 for b in bs))/len(bs) for k in QUANT}
        return dict(res=res, err=err, n=len(bs), names=[b.name for b in bs])

    def judge(q, delta, se_x, se_y=0.0):
        """(ratio, verdict): the difference in units of sigma_seed, ok within 2.8 sigma_seed; else by the quoted errors."""
        if q in sig and sig[q] > 0:
            return delta/sig[q], ('ok' if abs(delta) <= 2.8*sig[q] else 'DIFF')
        se = math.hypot(se_x, se_y)*infl(q)
        if not (se > 0):
            return float('nan'), '?'
        return delta/se, ('ok' if abs(delta) <= 2.8*se else 'DIFF')

    # --- per run: results.csv, blocks.csv
    allrows = entries + base_all
    with open(os.path.join(out, 'results.csv'), 'w') as f:
        f.write('id,stage,seed,theta,Re_c,e,Lx,Ly,Lz,t_end,done,tau_q2,' + ','.join(QUANT) + ',' + ','.join('se_' + k for k in QUANT)
                + ',' + ','.join(k + '_w2' for k in QUANT) + '\n')
        for e in allrows:
            r = e.run
            w2 = [e.res2[k] if e.res2 else float('nan') for k in QUANT]
            f.write(f'{r.name},{e.stage},{r.seed},{r.theta:.2f},{r.Re_c:.0f},{r.edge:g},{r.lx:.4f},{r.ly:.4f},{r.lz:.4f},{r.t[-1]:.1f},{int(r.done)},{e.tau:.3g},'
                    + ','.join(f'{e.res[k]:.6g}' for k in QUANT) + ',' + ','.join(f'{e.err[k]:.3g}' for k in QUANT)
                    + ',' + ','.join(f'{x:.6g}' for x in w2) + '\n')
    with open(os.path.join(out, 'blocks.csv'), 'w') as f:
        f.write('id,quantity,' + ','.join(f'se_block{b:g}' for b in BLOCKS) + ',mean_from%g,mean_from%g\n' % (a.transient, a.transient2))
        for e in allrows:
            se = [e.run.stats(a.transient, b)[1] for b in BLOCKS]
            for k in QUANT:
                f.write(f'{e.name},{k},' + ','.join(f'{s[k]:.3g}' for s in se) + f',{e.res[k]:.6g},{e.res2[k] if e.res2 else float("nan"):.6g}\n')
    print(f'{"run":20s} {"stage":5s} {"t":>6s} {"tau":>5s}  ' + '  '.join(f'{LABEL[k][:9]:>9s}' for k in ('q2n', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v', 'E_low')))
    for e in sorted(entries, key=lambda e: (e.stage, e.name)):
        print(f'{e.name:20s} {e.stage:5s} {e.run.t[-1]:6.0f} {e.tau:5.1f}  ' + '  '.join(
            f'{e.res[k]:9.4f}' for k in ('q2n', 'Sstar', 'uv', 'Leps', 'b_par', 'b_v', 'E_low')) + ('' if e.run.done else '  (running)'))
    print('(results.csv, blocks.csv: the errors against the block length and the window check)\n')

    # --- 1. box convergence
    rows = []
    for e in entries:
        if e.stage not in ('1a', '1b', '1c', '1e'):
            continue
        b = base_at(e.run.theta)
        if b is None:
            rows.append((e.name, e.run.theta, None)); continue
        rows.append((e.name, e.run.theta, {q: (e.res[q], e.err[q], b['res'][q], b['err'][q], *judge(q, e.res[q] - b['res'][q], e.err[q], b['err'][q]))
                                           for q in BOXQ}))
    if rows:
        print('BOX CONVERGENCE  (difference to the baseline at the same angle in units of sigma_seed; * = beyond 2.8)')
        print(f'{"run":16s} {"theta":>5s}  ' + ' '.join(f'{LABEL[q][:10]:>10s}' for q in BOXQ))
        with open(os.path.join(out, 'box.csv'), 'w') as f:
            f.write('id,theta,quantity,value,se,baseline,baseline_se,ratio_to_sigma,verdict\n')
            for name, th, d in rows:
                if d is None:
                    print(f'{name:16s} {th:5.1f}  no baseline run at this angle'); continue
                print(f'{name:16s} {th:5.1f}  ' + ' '.join(f'{d[q][4]:9.2f}{"*" if d[q][5] == "DIFF" else " "}' for q in BOXQ))
                for q in BOXQ:
                    v = d[q]
                    f.write(f'{name},{th},{q},{v[0]:.6g},{v[1]:.3g},{v[2]:.6g},{v[3]:.3g},{v[4]:.3g},{v[5]}\n')
        print('(box.csv)\n')

    # --- 2. orientation: x(30) - x(0) in a family
    fam = {}
    for e in entries:
        fam.setdefault(re.sub(r'-t\d+$', '', e.name) if e.stage != '1d' else '1d-B', {})[round(e.run.theta, 1)] = e
    b0, b30 = base_at(0.0), base_at(30.0)
    orows = []
    for key, d in sorted(fam.items()):
        ref = None
        if 0.0 in d:
            ref = (d[0.0].res, d[0.0].err, d[0.0].name)
        elif key == '1d-B' and b0:
            ref = (b0['res'], b0['err'], 'baseline')
        for th, e in sorted(d.items()):
            if th == 0.0 or ref is None:
                continue
            for q in BOXQ + ['uv']:
                orows.append((key, th, q, e.res[q] - ref[0][q], *judge(q, e.res[q] - ref[0][q], e.err[q], ref[1][q]), ref[2], e))
    if b0 and b30:      # the reference: the baseline's own 30 - 0
        for q in BOXQ + ['uv']:
            orows.append(('baseline', 30.0, q, b30['res'][q] - b0['res'][q], *judge(q, b30['res'][q] - b0['res'][q], b30['err'][q], b0['err'][q]), 'baseline', None))
    if orows:
        print('ORIENTATION  (x(theta) - x(0 deg) in units of sigma_seed; * = beyond 2.8) and the cross terms at theta > 0')
        keys = BOXQ + ['uv']
        print(f'{"family":20s} {"theta":>5s}  ' + ' '.join(f'{LABEL[q][:9]:>9s}' for q in keys) + '   <u_perp v>/q2    rho_par,perp')
        fams = []
        for r in orows:
            if (r[0], r[1]) not in fams:
                fams.append((r[0], r[1]))
        with open(os.path.join(out, 'orientation.csv'), 'w') as f:
            f.write('family,theta,quantity,difference,ratio_to_sigma,verdict,reference\n')
            for key, th in fams:
                rs = {r[2]: r for r in orows if (r[0], r[1]) == (key, th)}
                ent = rs['uv'][7]
                cross = ''
                if ent is not None:
                    for q in ('uperpv', 'rho_pp'):
                        x = ent.res[q]; ratio, v = judge(q, x, ent.err[q])
                        cross += f'  {x:+8.4f}{"*" if v == "DIFF" else " "}   '
                        f.write(f'{key},{th},{q}@theta,{x:.6g},{ratio:.3g},{v},zero\n')
                print(f'{key:20s} {th:5.1f}  ' + ' '.join(f'{rs[q][4]:8.2f}{"*" if rs[q][5] == "DIFF" else " "}' for q in keys) + cross)
                for q in keys:
                    r = rs[q]
                    f.write(f'{key},{th},{q},{r[3]:.6g},{r[4]:.3g},{r[5]},{r[6]}\n')
        print('(orientation.csv)\n')

    # --- 3. box-cycle resonance
    print(f'BOX-CYCLE RESONANCE  (power in the <v^2>(t) spectrum at k/T over its neighbours; * = above {a.line_ratio:g}; t >= {a.transient:g})')
    with open(os.path.join(out, 'resonance.csv'), 'w') as f:
        f.write('id,axis,period,harmonic,frequency,power_ratio,segments,flag\n')
        for e in entries:
            sp = e.run.vv_spectrum(a.transient)
            if sp is None:
                print(f'{e.name:20s} too short for a spectrum'); continue
            freq, pw, nseg = sp
            parts = []
            for axis, T in e.run.box_periods():
                for k in (1, 2, 3, 4):
                    fk = k/T
                    ratio = line_ratio(freq, pw, fk)
                    flag = 'LINE' if ratio > a.line_ratio else ''
                    f.write(f'{e.name},{axis},{T:.4f},{k},{fk:.4f},{ratio:.3g},{nseg},{flag}\n')
                    parts.append(f'{axis}{k}:{ratio:4.1f}{"*" if flag else " "}')
            print(f'{e.name:20s} periods ' + ' '.join(f'{ax}={T:.2f}' for ax, T in e.run.box_periods()) + f'  ({nseg} segments)  ' + ' '.join(parts))
    print('(resonance.csv)\n')

    # --- 4. decorrelation from the column spectra
    print(f'BOX DECORRELATION  (R(dx,0) and R(0,dz) of the column spectra at half the box; * = |R| > {a.decor_tol:g}; max over a quarter..half box)')
    with open(os.path.join(out, 'decorrelation.csv'), 'w') as f:
        f.write('id,Lx,Lz,R_half_x,R_half_z,maxabs_x,maxabs_z,files,flag\n')
        for e in entries:
            sp = read_spectra(e.run.dir, a.transient)
            if sp is None:
                print(f'{e.name:20s} no column spectra after t = {a.transient:g}'); continue
            kx, kz, E, nf = sp
            r = e.run
            gx = np.linspace(r.lx/4, r.lx/2, 25); gz = np.linspace(r.lz/4, r.lz/2, 25)
            cx = np.array([correlation(kx, kz, E, x, 0.0) for x in gx]); cz = np.array([correlation(kx, kz, E, 0.0, z) for z in gz])
            flag = 'FLAG' if max(abs(cx[-1]), abs(cz[-1])) > a.decor_tol else ''
            f.write(f'{e.name},{r.lx:.4f},{r.lz:.4f},{cx[-1]:.4f},{cz[-1]:.4f},{abs(cx).max():.4f},{abs(cz).max():.4f},{nf},{flag}\n')
            print(f'{e.name:20s} R(Lx/2) {cx[-1]:+.3f}{"*" if abs(cx[-1]) > a.decor_tol else " "}  R(Lz/2) {cz[-1]:+.3f}{"*" if abs(cz[-1]) > a.decor_tol else " "}'
                  f'   max|R| {abs(cx).max():.3f} {abs(cz).max():.3f}   ({nf} spectra files)')
    print('(decorrelation.csv)\n')

    # --- 5. edge
    by_id = {e.name: e for e in entries}
    st2 = [e for e in entries if e.stage == '2']
    if st2:
        print('EDGE  (quantities against e; e = 0.3 from the baseline (A_h = 6.06) and stage 1a (A_h = 4.5); interaction = [x(0.5) - x(0.15)]_6.06 - [..]_4.5)')
        with open(os.path.join(out, 'edge.csv'), 'w') as f:
            f.write('box,theta,quantity,e015,e015_se,e030,e030_se,e050,e050_se,interaction,interaction_se,z\n')
            for th, tag in ((0.0, 't00'), (30.0, 't30')):
                data = {}
                for box, lab, ref in (('B', '6.06', base_at(th)), ('H45', '4.5', by_id.get(f'1a-H45-{tag}'))):
                    lo, hi = by_id.get(f'2-{box}-e015-{tag}'), by_id.get(f'2-{box}-e050-{tag}')
                    if isinstance(ref, Entry):
                        ref = dict(res=ref.res, err=ref.err)
                    if lo and hi:
                        data[box] = (lab, lo, ref, hi)
                for box, (lab, lo, ref, hi) in data.items():
                    print(f'  A_h = {lab:5s} theta {th:4.1f}:  e =        0.15            0.30            0.50')
                    for q in EDGEQ:
                        mid = f'{ref["res"][q]:8.4f}({ref["err"][q]:.4f})' if ref else '       -       '
                        print(f'      {LABEL[q][:16]:16s}  {lo.res[q]:8.4f}({lo.err[q]:.4f})  {mid}  {hi.res[q]:8.4f}({hi.err[q]:.4f})')
                def interaction(q):
                    """[x(0.5) - x(0.15)] at 6.06 minus the same at 4.5, its error and z."""
                    B, H = data['B'], data['H45']
                    inter = (B[3].res[q] - B[1].res[q]) - (H[3].res[q] - H[1].res[q])
                    se = infl(q)*math.sqrt(B[3].err[q]**2 + B[1].err[q]**2 + H[3].err[q]**2 + H[1].err[q]**2)
                    return inter, se, (inter/se if se > 0 else float('nan'))
                if len(data) == 2:
                    print(f'  interaction at theta {th:g} (z = difference/error):  ' + '  '.join(
                        f'{LABEL[q][:9]} {interaction(q)[2]:+.1f}' for q in EDGEQ))
                for box, (lab, lo, ref, hi) in data.items():
                    for q in EDGEQ:
                        inter, se, z = interaction(q) if len(data) == 2 else (float('nan'),)*3
                        mid, mse = (ref['res'][q], ref['err'][q]) if ref else (float('nan'), float('nan'))
                        f.write(f'{lab},{th},{q},{lo.res[q]:.6g},{lo.err[q]:.3g},{mid:.6g},{mse:.3g},{hi.res[q]:.6g},{hi.err[q]:.3g},{inter:.4g},{se:.3g},{z:.3g}\n')
        print('(edge.csv)\n')

    # --- 6. Reynolds number
    st3 = [e for e in entries if e.stage in ('3a', '3b')]
    if st3:
        print('REYNOLDS NUMBER  (trend, not pass or fail; q2 level against the Re_c = 3000 run of the same box and angle)')
        print(f'{"run":20s} {"box":>5s} {"theta":>5s} {"Re_c":>6s} {"t":>6s} ' + ' '.join(f'{LABEL[q][:10]:>10s}' for q in REYQ) + '  flow')
        with open(os.path.join(out, 'reynolds.csv'), 'w') as f:
            f.write('id,box,theta,Re_c,t_end,' + ','.join(REYQ) + ',q2_last50,q2_ratio,flow\n')
            refs = []
            for e in st3:
                box = 'B' if abs(e.run.lx - 10.5) < 1e-6 else 'H45'
                th = round(e.run.theta, 1)
                refe = base_at(th) if box == 'B' else by_id.get(f'1a-H45-t{int(th):02d}')
                if isinstance(refe, Entry):
                    refe = dict(res=refe.res, err=refe.err)
                refs.append((e, box, th, refe))
            seen = set()
            for e, box, th, refe in sorted(refs, key=lambda x: (x[1], x[2], x[0].run.Re_c)):
                if refe is not None and (box, th) not in seen:     # the Re_c = 3000 reference of this box and angle, once
                    seen.add((box, th))
                    print(f'{"(reference)":20s} {box:>5s} {th:5.1f} {3000:6d} {"":>6s} ' + ' '.join(f'{refe["res"][q]:10.4f}' for q in REYQ) + '  turbulent')
                    f.write(f'reference,{box},{th},3000,,' + ','.join(f'{refe["res"][q]:.6g}' for q in REYQ) + ',,1,turbulent\n')
                now, before = e.run.q2_level()
                ratio = now/(refe['res']['q2n']*(e.run.S0*e.run.lam)**2) if refe else float('nan')
                flow = 'turbulent'
                if ratio < 0.05:
                    flow = 'LAMINAR (q2 < 5 % of the reference' + (', still falling)' if now < before else ')')
                elif now < 0.5*(refe['res']['q2n']*(e.run.S0*e.run.lam)**2 if refe else now):
                    flow = 'weak (q2 < 50 % of the reference)'
                print(f'{e.name:20s} {box:>5s} {th:5.1f} {e.run.Re_c:6.0f} {e.run.t[-1]:6.0f} ' + ' '.join(f'{e.res[q]:10.4f}' for q in REYQ) + f'  {flow}'
                      + ('' if e.run.done else ' (running)'))
                f.write(f'{e.name},{box},{th},{e.run.Re_c:.0f},{e.run.t[-1]:.1f},' + ','.join(f'{e.res[q]:.6g}' for q in REYQ) + f',{now:.6g},{ratio:.4g},{flow}\n')
        print('(reynolds.csv)\n')
    print(f'csv files in {os.path.relpath(out, HERE)}/')


if __name__ == '__main__':
    main()
