#!/usr/bin/env python3
"""Write the decks of the campaign: runs/<case>/theta_<deg>/hst.in from template.in.

    python3 make_runs.py            # (re)generates every deck, plain Python 3, no packages

The shear has magnitude 1 and direction theta: S = cos(theta), s2_amplitude =
sin(theta), constant.  Edit CASES / ANGLES below to change the campaign; the
box, grid, Re and run length are in template.in and are the same for all runs.
"""
import math
import os

EDGE = 0.3                                  # prod_edge: smooth edge, half width 0.3 k_c
ANGLES = [0.0, 15.0, 30.0, 45.0]            # deg

# case name -> k_c and the initial condition (&init of the deck)
# The defaults are those of the campaign at k_c = 2 pi/sqrt 3 (doc.production).
# If a case only becomes turbulent from a different initial field, change
# amplitude / kpeak / seed here and rerun this script.
CASES = {
    "smooth_kc2.09": dict(kc=2 * math.pi / 3, amplitude=6.0e-4, kpeak=8.0, seed=1),
    "smooth_kc6.28": dict(kc=2 * math.pi,     amplitude=6.0e-4, kpeak=8.0, seed=1),
}

here = os.path.dirname(os.path.abspath(__file__))
template = open(os.path.join(here, "template.in")).read()

for case, c in CASES.items():
    for theta in ANGLES:
        th = math.radians(theta)
        text = template
        for key, val in {
            "@CASE@": case,
            "@EDGE@": repr(EDGE),
            "@KC@": repr(c["kc"]),
            "@THETA@": "%.1f" % theta,
            "@S@": repr(math.cos(th)),
            "@S2@": repr(math.sin(th)),
            "@AMPLITUDE@": "%.6f" % c["amplitude"],
            "@KPEAK@": repr(c["kpeak"]),
            "@SEED@": str(c["seed"]),
        }.items():
            text = text.replace(key, val)
        assert "@" not in text, "unreplaced placeholder"
        d = os.path.join(here, "runs", case, "theta_%04.1f" % theta)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "hst.in"), "w") as f:
            f.write(text)
        print("wrote", os.path.relpath(os.path.join(d, "hst.in"), here))
