#!/usr/bin/env python3
"""Per-column component spectra from the velocity snapshots (run where the fields are).

   python3 tools/compspec_extract.py OUTDIR RUN [RUN ...] [first snapshot index, default 3]

For each run the snapshots fields/field<i>.fld, i >= first, are read (CPL
layout, hst_io.f90: (0..nx, -nz..nz, ny + 4 rows, (u, w, v)) complex, C
order, after a text header) and the y-averages of |u|^2, |v|^2, |w|^2 and
Re(u conj(w)), Re(u conj(v)), Re(w conj(v)) per (kx, kz) column, averaged over the snapshots, are saved
as OUTDIR/<case>_<theta>.npz (arrays uu, vv, ww, uw, uv, wv of shape (nx+1, 2nz+1);
kx = alfa0*ix, kz = beta0*(iz - nz); the half lattice kx >= 0, not doubled).
"""
import glob, os, re, sys
import numpy as np
out = sys.argv[1]; args = sys.argv[2:]; first = 3
if args[-1].isdigit(): first = int(args.pop())
os.makedirs(out, exist_ok=True)
for run in args:
    acc = None; n = 0; times = []
    for f in sorted(glob.glob(run + '/fields/field*.fld'), key=lambda s: int(re.findall(r'field(\d+)', s)[0])):
        if int(re.findall(r'field(\d+)', f)[0]) < first: continue
        head = open(f, 'rb').read(600).decode('latin1')
        nx, nzs, nyc = (int(re.search(k + r'=(\d+)', head).group(1)) for k in ('nx', 'ny', 'nz'))   # CPL names: ny = our nz, nz = our ny + 1
        a0, b0 = (float(re.search(k + r'=([-0-9.eE+]+)', head).group(1)) for k in ('alpha0', 'beta0'))
        times.append(float(re.search(r'\bt_field=([-0-9.eE+]+)', head).group(1)))
        shape = (nx + 1, 2*nzs + 1, nyc + 3, 3)
        V = np.memmap(f, dtype='<c16', mode='r', offset=os.path.getsize(f) - 16*int(np.prod(shape)), shape=shape)[:, :, 2:-2, :]
        u, w, v = V[..., 0], V[..., 1], V[..., 2]
        s = np.stack([(abs(u)**2).mean(-1), (abs(v)**2).mean(-1), (abs(w)**2).mean(-1), (u*np.conj(w)).real.mean(-1),
                      (u*np.conj(v)).real.mean(-1), (w*np.conj(v)).real.mean(-1)])
        acc = s if acc is None else acc + s; n += 1
    acc /= n
    name = '_'.join(run.rstrip('/').split('/')[-2:])
    np.savez(os.path.join(out, name + '.npz'), uu=acc[0], vv=acc[1], ww=acc[2], uw=acc[3], uv=acc[4], wv=acc[5], alfa0=a0, beta0=b0, times=times)
    m = np.ones(nx + 1)*2; m[0] = 1
    print(name, n, 'snapshots, t', times[0], '..', times[-1], ' uu vv ww uw:', *('%.5f' % (acc[i]*m[:, None]).sum() for i in range(4)), flush=True)
