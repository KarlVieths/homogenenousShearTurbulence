#!/usr/bin/env python3
"""Write manifest.csv: one row per run of stages 1a-3b of the robustness campaign.

    python3 make_manifest.py                       # manifest.csv for the planned box (stage 0 is campaigns/square_seed_ensemble)
    python3 make_manifest.py --adopt 4.5 2 4.5     # stages 1d, 2-B, 3a, 3b-B regenerated for another adopted box
    python3 make_manifest.py --replicates 2        # a second seed per point in stages 1-2 (stage 0 gave R >= 2)
    python3 make_manifest.py --check manifest_document.csv      # compare with the manifest of the planning document

Every deck follows from Re_c, the three aspect ratios (box side / lambda_c), e and theta.  The filter is
fixed at lambda_c = 2 pi/k_c = sqrt 3, so Re_c is set through nu, i.e. `re`:

    L_i = A_ic lambda_c,   re = 1/nu = Re_c/(S_tot lambda_c^2) = Re_c/3,   n_i = round(d_i r L_i),
    r = max(1, (Re_c/3000)^(3/4)),   d = (16, 47.5, 24) modes per unit length in x, y, z.

The baseline box is the square box of the seed ensemble, Lx = Lz = 10.5 exactly (A_h = 6.06), Ly = 2 sqrt 3.
The mesh, the shear (S = cos theta, s2_amplitude = sin theta, |S_tot| = 1), the lengths and re are written in
full precision; the document's manifest has ten decimals (--check compares with a tolerance).

The cost columns are estimates (the calibration is at the top; recalibrate it from the in-progress run):
wall hours on the listed number of A100s = steps x 4 nx ny nz x NS_POINT / gpus / EFF[gpus], steps = t_max/dt,
dt = DT0/r; memory 0.30 GB per million of 4 nx ny nz (measured: examples/bench_1024.in, 20 steps, 39.3 GB per GPU on 8 GPUs = 0.294 GB per million;
the device-memory leak of binaries before 2026-11-25, ~27 KiB per step, is on top: 8-13 GiB over a run); disk 15 files (14 snapshots + the restart file) of
0.7 GB x nx ny nz/(168 x 165 x 252) per run.
"""
import argparse
import csv
import math
import os
import sys

LAMBDA_C = math.sqrt(3.0)               # 2 pi/k_c, prod_kc = 3.6275987285
D = (16.0, 47.5, 24.0)                  # modes per unit length in x, y, z (the 15 x 2 x 5 deck: 240, 95, 120)
BASE_H = 6.06                           # label of the baseline's Lx = Lz = 10.5 (= 6.0622 lambda_c)
T_MAX = 1400.0

NS_POINT = 4.85e-9                      # s per point per step on one A100 (planning document)
DT0 = 0.0046                            # time step at the baseline spacing
EFF = {1: 1.0, 2: 0.85, 4: 0.74}        # multi-GPU efficiency: the document's 31 h wall for 3b-B on four A100s
MEM_GB_PER_MPT = 0.30                   # 22 GB for 73.3 M points, 40 GB for 133 M (4 nx ny nz)
SNAP_GB = 0.7                           # one velocity snapshot at the baseline mesh
SNAP_COUNT = 14                         # dt_field = 100 over t_max = 1400

COLUMNS = ['id', 'stage', 'theta_deg', 'A_xc', 'A_yc', 'A_zc', 'Re_c', 'e', 'Lx', 'Ly', 'Lz', 'nx', 'ny', 'nz',
           're', 'S', 's2_amplitude', 'seed', 'gpus', 'est_wall_h', 'est_mem_gb', 'est_disk_gb']


def length(a):
    """Box side for an aspect ratio a = L/lambda_c; the baseline label gives 10.5 exactly."""
    return 10.5 if abs(a - BASE_H) < 1e-9 else a*LAMBDA_C


def fft_fit(n):
    while n % 2 == 0:
        n //= 2
    return n in (1, 3)


def dealiased(n):
    while not fft_fit(n):
        n += 1
    return n


def row(rid, stage, theta, A, re_c, e, seed, gpus=1, r_override=None):
    """One run: A = (A_xc, A_yc, A_zc), the aspect ratios (box side / lambda_c)."""
    L = [length(a) for a in A]
    r = r_override if r_override else max(1.0, (re_c/3000.0)**0.75)
    n = [int(d*r*l + 0.5) for d, l in zip(D, L)]
    th = math.radians(theta)
    S, S2 = (1.0, 0.0) if theta == 0 else (math.cos(th), math.sin(th))
    pts = 4.0*n[0]*n[1]*n[2]
    steps = T_MAX/(DT0/r)
    wall = steps*pts*NS_POINT/gpus/EFF[gpus]/3600.0
    snap = SNAP_GB*n[0]*n[1]*n[2]/(168.0*165*252)
    return dict(id=rid, stage=stage, theta_deg=theta, A_xc=A[0], A_yc=A[1], A_zc=A[2], Re_c=re_c, e=e,
                Lx=L[0], Ly=L[1], Lz=L[2], nx=n[0], ny=n[1], nz=n[2], re=re_c/(LAMBDA_C**2), S=S, s2_amplitude=S2,
                seed=seed, gpus=gpus, est_wall_h=wall, est_mem_gb=MEM_GB_PER_MPT*pts/1e6,
                est_disk_gb=(SNAP_COUNT + 1)*snap)


def check_ranks(rw):
    """The number of x-z pencils (= ranks of a run on one node) must divide nx+1 and nzd (src/hst_mpi.f90)."""
    g = rw['gpus']
    nzd = dealiased(3*rw['nz'])
    if g > 1 and ((rw['nx'] + 1) % g or nzd % g):
        raise SystemExit(f"{rw['id']}: {g} ranks do not divide nx+1 = {rw['nx'] + 1} and nzd = {nzd}")


def stages(adopt, replicates):
    """The runs of stages 1a-3b in the order of the planning document (seeds 201, 202, ...)."""
    BASE = (BASE_H, 2.0, BASE_H)
    H45 = (4.5, 2.0, 4.5)
    B = adopt                                    # the box stages 1d, 2, 3 use: the baseline unless stage 1 adopts another
    spec = []                                    # (id, stage, theta, A, Re_c, e, gpus, r_override)
    for tag, a in (('H3', 3.0), ('H45', 4.5), ('H9', 9.0)):
        for th in (0, 30):
            spec.append((f'1a-{tag}-t{th:02d}', '1a', th, (a, 2.0, a), 3000, 0.3, 1, None))
    for tag, ay in (('Y15', 1.5), ('Y3', 3.0)):
        for th in (0, 30):
            spec.append((f'1b-{tag}-t{th:02d}', '1b', th, (BASE_H, ay, BASE_H), 3000, 0.3, 1, None))
    for th in (0, 30):
        spec.append((f'1c-R-t{th:02d}', '1c', th, (9.0, 2.0, 6.0), 3000, 0.3, 1, None))
    for th, tag in ((22.5, 't225'), (37.5, 't375')):
        spec.append((f'1d-B-{tag}', '1d', th, B, 3000, 0.3, 1, None))
    spec.append(('1e-B-res-t30', '1e', 30, BASE, 3000, 0.3, 1, 1.5))
    for box, name in ((B, 'B'), (H45, 'H45')):
        for e, etag in ((0.15, 'e015'), (0.5, 'e050')):
            for th in (0, 30):
                spec.append((f'2-{name}-{etag}-t{th:02d}', '2', th, box, 3000, e, 1, None))
    for rc in (1000, 1500, 2000):
        spec.append((f'3a-B-Rc{rc}-t00', '3a', 0, B, rc, 0.3, 1, None))
    for box, name, g in ((B, 'B', 4), (H45, 'H45', 1)):
        for th in (0, 30):
            spec.append((f'3b-{name}-Rc6000-t{th:02d}', '3b', th, box, 6000, 0.3, g, None))
    rows = []
    for k, (rid, st, th, A, rc, e, g, ro) in enumerate(spec):
        rows.append(row(rid, st, th, A, rc, e, 201 + k, g, ro))
    # second seeds for the stages the error-bar factor R concerns (1-2): -r2 with seed + 100, -r3 with + 200, ...
    extra = []
    for rep in range(2, replicates + 1):
        for rw in rows:
            if rw['stage'] in ('1a', '1b', '1c', '1d', '1e', '2'):
                x = dict(rw); x['id'] = rw['id'] + f'-r{rep}'; x['seed'] = rw['seed'] + 100*(rep - 1)
                extra.append(x)
    return rows + extra


def write(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(COLUMNS)
        for rw in rows:
            out = []
            for c in COLUMNS:
                v = rw[c]
                if c in ('est_wall_h', 'est_mem_gb', 'est_disk_gb'):
                    out.append('%.3g' % v if v < 100 else '%.0f' % v)
                elif isinstance(v, float):
                    out.append(('%d' % v) if v == int(v) and abs(v) < 1e6 else repr(v))
                else:
                    out.append(str(v))
            w.writerow(out)


def check(rows, other):
    """Compare id, mesh, seed and gpus exactly and the lengths, re, S, s2 within 1e-9 with another manifest."""
    ref = {r['id']: r for r in csv.DictReader(open(other))}
    mine = {r['id']: r for r in rows}
    bad = 0
    for rid in sorted(set(ref) | set(mine)):
        if rid not in ref or rid not in mine:
            print(f'{rid}: only in {"this" if rid in mine else other}'); bad += 1; continue
        a, b = mine[rid], ref[rid]
        for c in ('stage', 'nx', 'ny', 'nz', 'seed', 'gpus'):
            if str(a[c]) != b[c]:
                print(f'{rid}: {c} {a[c]} != {b[c]}'); bad += 1
        for c in ('theta_deg', 'A_xc', 'A_yc', 'A_zc', 'Re_c', 'e', 'Lx', 'Ly', 'Lz', 're', 'S', 's2_amplitude'):
            if abs(float(a[c]) - float(b[c])) > 1e-9*max(1.0, abs(float(b[c]))):
                print(f'{rid}: {c} {a[c]} != {b[c]}'); bad += 1
        if abs(float(a['est_wall_h']) - float(b['est_wall_h'])) > 0.1*float(b['est_wall_h']) + 0.05:
            print(f'{rid}: est_wall_h {float(a["est_wall_h"]):.1f} against {b["est_wall_h"]} (the cost model, not the deck)')
    print('check:', 'identical to ' + other + ' (decks)' if not bad else f'{bad} differences')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adopt', nargs=3, type=float, metavar=('A_XC', 'A_YC', 'A_ZC'), default=(BASE_H, 2.0, BASE_H),
                    help='aspect ratios (box side / lambda_c) of the box stage 1 adopts; 6.06 is the 10.5 box')
    ap.add_argument('--replicates', type=int, default=1, help='seeds per point in stages 1-2 (default 1)')
    ap.add_argument('--out', default='manifest.csv')
    ap.add_argument('--check', help='compare with another manifest (the document\'s) and exit with 1 on differences')
    a = ap.parse_args()
    rows = stages(tuple(a.adopt), a.replicates)
    for rw in rows:
        check_ranks(rw)
    ids = [r['id'] for r in rows]; seeds = [r['seed'] for r in rows]
    assert len(set(ids)) == len(ids) and len(set(seeds)) == len(seeds), 'ids and seeds must be unique'
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    write(rows, out)
    gpuh = sum(r['est_wall_h']*r['gpus'] for r in rows)
    print(f'{len(rows)} runs, about {gpuh:.0f} A100 GPU-hours, {sum(r["est_disk_gb"] for r in rows)/1000:.2f} TB of output '
          f'(snapshots + restart files); {os.path.relpath(out, here)} written')
    if a.check:
        sys.exit(1 if check(rows, a.check) else 0)


if __name__ == '__main__':
    main()
