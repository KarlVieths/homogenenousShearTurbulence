#!/usr/bin/env python3
"""Write the decks of the seed ensemble: runs/square_th30_s<seed>/hst.in from template.in.

    python3 make_runs.py            # (re)generates every deck, plain Python 3, no packages

Four runs of one case that differ only in `seed`, the seed of the random initial
field: square box 10.5 x 2 sqrt 3 x 10.5, smooth production filter (edge 0.3) at
k_c = 2 pi/sqrt 3, shear direction 30 deg, Re = 1000.  The shear has magnitude 1
and direction theta: S = cos(theta), s2_amplitude = sin(theta), constant.
"""
import math
import os

THETA = 30.0                                # deg
EDGE = 0.3                                  # prod_edge
KC = 2 * math.pi / math.sqrt(3)             # prod_kc, l_f = 2 pi/k_c = sqrt 3
LX = 10.5                                   # Lx = Lz
LY = 2 * math.sqrt(3)                       # 2 l_f
SEEDS = [101, 102, 103, 104]                # any integers except the seeds of existing runs (the default deck: 1)
AMPLITUDE = 7.0e-4                          # &init amplitude, per mode; see README, "Initial field"
KPEAK = 8.0

here = os.path.dirname(os.path.abspath(__file__))
template = open(os.path.join(here, "template.in")).read()
assert len(set(SEEDS)) == len(SEEDS) and 1 not in SEEDS, "seeds must be distinct and differ from the default seed 1"

th = math.radians(THETA)
for seed in SEEDS:
    case = "square_th%02d_s%d" % (round(THETA), seed)
    text = template
    for key, val in {
        "@CASE@": case,
        "@EDGE@": repr(EDGE),
        "@KC@": repr(KC),
        "@THETA@": "%.1f" % THETA,
        "@S@": repr(math.cos(th)),
        "@S2@": repr(math.sin(th)),
        "@ALFA0@": repr(2 * math.pi / LX),
        "@LY@": repr(LY),
        "@AMPLITUDE@": "%.6g" % AMPLITUDE,
        "@KPEAK@": repr(KPEAK),
        "@SEED@": str(seed),
    }.items():
        text = text.replace(key, val)
    assert "@" not in text, "unreplaced placeholder"
    d = os.path.join(here, "runs", case)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "hst.in"), "w") as out:
        out.write(text)
    print("wrote", os.path.relpath(os.path.join(d, "hst.in"), here))
