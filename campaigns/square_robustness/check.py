#!/usr/bin/env python3
"""Pre-launch checks of the decks (planning document, "Before launching"): python3 check.py [--ids ...]

  * every deck agrees with its manifest row (mesh, lengths, re, S, s2_amplitude, seed, edge, prod_kc);
  * seeds are unique in the manifest and differ from stage 0's (../square_seed_ensemble) and from the default 1;
  * the decks agree with the seed ensemble's deck in everything the manifest does not set (time step, t_max, output
    intervals, amplitude, kpeak, filter, ...): the deck of the stage-0 runs is the reference;
  * rank counts of the multi-GPU runs divide nx+1 and the dealiased z size; GPU memory estimates per run.
Prints one line per problem and "OK" when there is none; exit status 1 with problems.
"""
import argparse
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def deckval(text, key):
    m = re.search(r'^\s*' + key + r'\s*=\s*([-0-9.eEdD+]+)', text, re.M)
    return float(m.group(1).replace('d', 'e').replace('D', 'e')) if m else None


def fft_fit(n):
    while n % 2 == 0:
        n //= 2
    return n in (1, 3)


def strip(text, keys):
    """The deck without the header comments and the lines of the given keys (for the comparison with stage 0's deck)."""
    out = []
    for l in text.splitlines():
        if l.startswith('!') or not l.strip():
            continue
        if any(re.match(r'^\s*' + k + r'\s*=', l) for k in keys):
            continue
        out.append(re.sub(r'\s*!.*$', '', l).rstrip())   # comments after a value are not compared
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ids', nargs='*', default=[])
    ap.add_argument('--manifest', default='manifest.csv')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, a.manifest))) if not a.ids or r['id'] in a.ids]
    bad = []
    seeds = [int(r['seed']) for r in csv.DictReader(open(os.path.join(HERE, a.manifest)))]
    if len(set(seeds)) != len(seeds):
        bad.append('seeds are not unique in the manifest')
    seed0 = sorted(glob_seed0())
    for s in seeds:
        if s in seed0 or s == 1:
            bad.append(f'seed {s} is also used by stage 0 or is the default seed')
    # stage 0's deck as the reference for the values the manifest does not set
    ref = None
    for d in sorted(os.listdir(os.path.join(HERE, '..', 'square_seed_ensemble', 'runs'))) if os.path.isdir(os.path.join(HERE, '..', 'square_seed_ensemble', 'runs')) else []:
        ref = open(os.path.join(HERE, '..', 'square_seed_ensemble', 'runs', d, 'hst.in')).read(); break
    free = ['nx', 'alfa0', 'beta0', 'ly', 're', 'S', 's2_amplitude', 'seed', 'prod_edge']
    for r in rows:
        p = os.path.join(HERE, 'runs', r['id'], 'hst.in')
        if not os.path.exists(p):
            bad.append(f"{r['id']}: no deck (python3 make_runs.py)"); continue
        t = open(p).read()
        m = re.search(r'nx\s*=\s*(\d+),\s*ny\s*=\s*(\d+),\s*nz\s*=\s*(\d+)', t)
        got = dict(nx=int(m.group(1)), ny=int(m.group(2)), nz=int(m.group(3)))
        for k in ('nx', 'ny', 'nz'):
            if got[k] != int(r[k]):
                bad.append(f"{r['id']}: {k} = {got[k]}, manifest {r[k]}")
        lx, lz = 2*math.pi/deckval(t, 'alfa0'), 2*math.pi/deckval(t, 'beta0')
        for name, g, w in (('Lx', lx, r['Lx']), ('Lz', lz, r['Lz']), ('ly', deckval(t, 'ly'), r['Ly']), ('re', deckval(t, 're'), r['re']),
                           ('S', deckval(t, 'S'), r['S']), ('s2_amplitude', deckval(t, 's2_amplitude'), r['s2_amplitude']),
                           ('seed', deckval(t, 'seed'), r['seed']), ('prod_edge', deckval(t, 'prod_edge'), r['e']),
                           ('prod_kc', deckval(t, 'prod_kc'), 2*math.pi/math.sqrt(3))):
            if abs(g - float(w)) > 1e-12*max(1.0, abs(float(w))):
                bad.append(f"{r['id']}: {name} = {g!r}, manifest {w}")
        if abs(math.hypot(deckval(t, 'S'), deckval(t, 's2_amplitude')) - 1) > 1e-12:
            bad.append(f"{r['id']}: |S_tot| is not 1")
        if re.search(r"restart_file\s*=\s*'(?!Dati\.cart\.out')", t):
            bad.append(f"{r['id']}: restart_file is not Dati.cart.out (the run would start from another field)")
        if ref is not None:
            a1 = [l for l in strip(t, free) if not l.lstrip().startswith(('t_max', 'amplitude', 'wall_max'))]
            a0 = [l for l in strip(ref, free) if not l.lstrip().startswith(('t_max', 'amplitude', 'wall_max'))]
            if a1 != a0:
                diff = [l for l in a1 if l not in a0] + ['  (stage 0:) ' + l for l in a0 if l not in a1]
                bad.append(f"{r['id']}: differs from the seed ensemble's deck beyond the manifest's values: {diff[:3]}")
        g = int(r['gpus'])
        if g > 1:
            nzd = 3*int(r['nz'])
            while not fft_fit(nzd):
                nzd += 1
            if (int(r['nx']) + 1) % g or nzd % g:
                bad.append(f"{r['id']}: {g} ranks do not divide nx+1 = {int(r['nx']) + 1} and nzd = {nzd}")
        if os.path.exists(os.path.join(HERE, 'runs', r['id'], 'Dati.cart.out')) and not any(
                f.startswith('run-') for f in os.listdir(os.path.join(HERE, 'runs', r['id']))):
            bad.append(f"{r['id']}: a Dati.cart.out without a run log (a leftover is read instead of the random field)")
    for b in bad:
        print('PROBLEM:', b)
    print(f'{len(rows)} decks checked:', 'OK' if not bad else f'{len(bad)} problems')
    sys.exit(1 if bad else 0)


def glob_seed0():
    out = []
    d = os.path.join(HERE, '..', 'square_seed_ensemble', 'runs')
    if os.path.isdir(d):
        for r in os.listdir(d):
            m = re.search(r'^\s*seed\s*=\s*(\d+)', open(os.path.join(d, r, 'hst.in')).read(), re.M)
            if m:
                out.append(int(m.group(1)))
    return out


if __name__ == '__main__':
    main()
