#!/usr/bin/env python3
"""Write the decks of the robustness campaign: runs/<id>/hst.in from template.in and manifest.csv.

    python3 make_runs.py                       # every run of the manifest
    python3 make_runs.py --stages 1a,1b        # only these stages
    python3 make_runs.py --ids 1a-H3-t00 ...   # only these runs
    python3 make_runs.py --tmax 2000           # longer runs (every deck; stage 0 gave R >= 2 and the answer is longer runs)
    python3 make_runs.py --init-q2 0.113       # amplitude per run such that the first q2 is 0.113 in every box

L, re, S and s2_amplitude, the mesh and the seed are those of manifest.csv (make_manifest.py writes it);
the other values are the seed ensemble's: prod_kc = 2 pi/sqrt 3, cflmax 0.7, t_max 1400, dt_stat 0.01,
dt_field = dt_save = 100, amplitude 7e-4, kpeak 8 (see template.in).
The first q2 of the random field is proportional to amplitude^2 Lx Lz and independent of Ly and Re
(measured with the code's generator on the 83-249 mode boxes: q2_0 = Q2_PER_A2_AREA amplitude^2 Lx Lz, within 3 %):
with the fixed amplitude it is 0.027 in the 3 lambda_c box and 0.25 in the 9 lambda_c box, against 0.113 in the
baseline.  --init-q2 sets the amplitude per run to give the same first q2 everywhere.  A deck of a run that has already
produced output (a log or a restart file) is not overwritten without --force: the job script edits it.
"""
import argparse
import csv
import glob
import math
import os

KC = 2 * math.pi / math.sqrt(3)             # prod_kc, lambda_c = 2 pi/k_c = sqrt 3
AMPLITUDE = 7.0e-4                          # &init amplitude (the seed ensemble's), per mode
Q2_PER_A2_AREA = 2099.0                     # first q2 = this x amplitude^2 x Lx x Lz (kpeak 8; the 10.5 box: 0.1134 at 7e-4)
KPEAK = 8.0
CFL = 0.7
T_MAX = 1400.0

here = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default='manifest.csv')
    ap.add_argument('--stages', default='', help='comma-separated stages (1a,1b,...); default all')
    ap.add_argument('--ids', nargs='*', default=[], help='only these run ids')
    ap.add_argument('--tmax', type=float, default=T_MAX)
    ap.add_argument('--amplitude', type=float, default=AMPLITUDE)
    ap.add_argument('--init-q2', type=float, default=0.0, help='scale the amplitude per run to this first q2 (default: the fixed amplitude)')
    ap.add_argument('--force', action='store_true', help='overwrite decks of runs that have output')
    a = ap.parse_args()
    template = open(os.path.join(here, 'template.in')).read()
    rows = list(csv.DictReader(open(os.path.join(here, a.manifest))))
    stages = [s for s in a.stages.split(',') if s]
    n = 0
    for r in rows:
        if (stages and r['stage'] not in stages) or (a.ids and r['id'] not in a.ids):
            continue
        d = os.path.join(here, 'runs', r['id'])
        if not a.force and (glob.glob(os.path.join(d, 'run-*.log')) or os.path.exists(os.path.join(d, 'Dati.cart.out'))):
            print(f"skip {r['id']}: has output (--force overwrites the deck)")
            continue
        amp = a.amplitude
        if a.init_q2 > 0:
            amp = math.sqrt(a.init_q2/(Q2_PER_A2_AREA*float(r['Lx'])*float(r['Lz'])))
        text = template
        for key, val in {
            '@ID@': r['id'], '@STAGE@': r['stage'], '@THETA@': r['theta_deg'], '@EDGE@': r['e'],
            '@AX@': r['A_xc'], '@AY@': r['A_yc'], '@AZ@': r['A_zc'], '@RE_C@': r['Re_c'],
            '@LX@': r['Lx'], '@LY@': r['Ly'], '@LZ@': r['Lz'],
            '@NX@': r['nx'], '@NY@': r['ny'], '@NZ@': r['nz'],
            '@ALFA0@': repr(2 * math.pi / float(r['Lx'])), '@BETA0@': repr(2 * math.pi / float(r['Lz'])),
            '@RE@': r['re'], '@S@': r['S'], '@S2@': r['s2_amplitude'],
            '@SEED@': r['seed'], '@AMPLITUDE@': '%.6g' % amp, '@KPEAK@': repr(KPEAK),
            '@KC@': repr(KC), '@CFL@': repr(CFL), '@TMAX@': repr(a.tmax),
        }.items():
            text = text.replace(key, val)
        assert '@' not in text, 'unreplaced placeholder'
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'hst.in'), 'w') as f:
            f.write(text)
        n += 1
    print(f'{n} decks written to runs/')


if __name__ == '__main__':
    main()
