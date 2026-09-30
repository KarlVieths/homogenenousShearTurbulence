#!/usr/bin/env python3
"""
Convert the new simulation input deck (hst.in, a Fortran namelist) into the
flat "name = value" token file (readinput.in) that the CPL post-processor
(post/cpl/readinput.cpl) reads.

WHY: the new simulator (src/hst_input.f90 / src/hst_io.f90) reads a namelist
(hst.in, groups &mesh / &physics / &time_control).  The CPL post-processor's
READ BY NAME (scanrec) only accepts flat `name = value` tokens -- it halts on
the `&group` / `/` Fortran namelist delimiters, so it cannot read hst.in
verbatim.  This script is the compatibility boundary: it pulls every value the
post-processor needs out of hst.in and emits them as flat tokens.

Axis relabel: hst.in &mesh uses the NEW-code convention
  ny = physical shear points, nz = spanwise Fourier modes,
while the CPL post-processor uses
  ny_cpl = spanwise modes (= nz_new),   nz_cpl = ny_new + 1.
The post-processor's array extents are COMPILE-TIME constants (CPL rejects
runtime assignment to an INTEGER CONSTANT), so it does NOT read nx/ny/nz from
the deck; this script only checks them against the compiled values (below) and
warns on mismatch.

Usage:
    python3 convert_hstin.py [hst.in] [readinput.in]
Defaults: input hst.in, output readinput.in (next to this script).
"""

import re
import sys
from pathlib import Path

# Compiled CPL-convention array extents (post/cpl/readinput.cpl).  Keep in
# sync with the `INTEGER CONSTANT nx=.., ny=.., nz=..` line there.
COMPILED_NX = 65
COMPILED_NY = 22   # spanwise modes            = hst.in &mesh nz
COMPILED_NZ = 87   # physical points + 1       = hst.in &mesh ny + 1

# Fortran logical -> the YES/NO spelling the CPL boolean reader understands.
BOOL = {'.true.': 'YES', '.false.': 'NO', 'true': 'YES', 'false': 'NO',
        't': 'YES', 'f': 'NO'}

# Order matters: readinput.cpl issues READ BY NAME in this exact order (the
# CPL file cursor advances forward).  (name, new_deck_key, group, mapper)
TOKENS = [
    ('alfa0',            'alfa0',          'mesh'),
    ('beta0',            'beta0',          'mesh'),
    ('ystretch',         'ystretch',       'mesh'),
    ('re',               're',             'physics'),
    ('S',                'S',              'physics'),
    ('s2_amplitude',     's2_amplitude',   'physics'),
    ('s2_period',        's2_period',      'physics'),
    ('s2_start',         's2_start',       'physics'),
    ('deltat',           'deltat',         'time_control'),
    ('cflmax',           'cflmax',         'time_control'),
    ('t_max',            't_max',          'time_control'),
    ('dt_stat',          'dt_stat',        'time_control'),
    ('dt_field',         'dt_field',       'time_control'),
    ('dt_save',          'dt_save',        'time_control'),
    ('time_from_restart','time_from_restart','time_control'),
    ('restart_file',     'restart_file',   'time_control'),
]

def parse_deck(path):
    """Return dict key -> (value_string, raw) for the given namelist deck."""
    vals = {}
    group = None
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.split('!', 1)[0].strip()   # drop trailing comments
        if not stripped:
            continue
        low = stripped.lower()
        if low.startswith('&'):
            group = low[1:].strip().rstrip('&').strip()
            continue
        if stripped == '/' or (low == '&end' and group):
            group = None
            continue
        m = re.match(r'([A-Za-z_]\w*)\s*=\s*(.+)', stripped)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val.endswith(','):
                val = val[:-1].strip()
            vals[key] = val
    return vals

def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('hst.in')
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('readinput.in')

    vals = parse_deck(src)

    # Optional: verify the compile-time mesh agrees with the deck (axis-relabelled).
    try:
        deck_nx = int(vals.get('nx'))
        deck_ny = int(vals.get('ny'))
        deck_nz = int(vals.get('nz'))
        if (deck_nx, deck_nz, deck_ny + 1) != (COMPILED_NX, COMPILED_NY, COMPILED_NZ):
            print(f'WARNING: postprocess is compiled for nx={COMPILED_NX} ny={COMPILED_NY} nz={COMPILED_NZ} '
                  f'(CPL convention) but {src.name} says &mesh nx={deck_nx} ny={deck_ny} nz={deck_nz}. '
                  'Recompile with matching constants or the field headers will not line up.',
                  file=sys.stderr)
    except (KeyError, ValueError):
        pass

    out = []
    for name, key, grp in TOKENS:
        if key not in vals:
            print(f'WARNING: {src.name} has no `{key}` in &{grp}; leaving it unset.', file=sys.stderr)
            continue
        val = vals[key]
        low = val.lower()
        if low in BOOL:
            val = BOOL[low]
        elif val.startswith("'") and val.endswith("'"):
            val = val[1:-1]                       # strip quotes for CPL string read
        out.append(f'{name} = {val}')

    dst.write_text('\n'.join(out) + '\n')
    print(f'Wrote {dst} ({len(out)} tokens) from {src}.')

if __name__ == '__main__':
    main()
