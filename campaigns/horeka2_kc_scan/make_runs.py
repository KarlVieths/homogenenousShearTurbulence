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

# case name -> k_c and the initial field.
# Either the random field of &init (amplitude, kpeak, seed; the defaults are
# those of the campaign at k_c = 2 pi/sqrt 3, doc.production), or a velocity
# field read from a file: set init_file to a path (absolute, or relative to the
# run directory) or to a function of theta that returns one; the file is read
# with the clock at 0 (time_from_restart = .false.), and must have this box
# (alfa0, beta0 equal) and may have another grid (README, "Initial field").
# init_file=None: no file, the random field.
def from_3_63(theta):          # the end of the run of the earlier campaign at the same angle
    return "/ABSOLUTE/PATH/TO/filter_smooth_kc3.63/theta_%04.1f/Dati.cart.out" % theta

CASES = {
    "smooth_kc2.09": dict(kc=2 * math.pi / 3, amplitude=6.0e-4, kpeak=8.0, seed=1, init_file=None),
    "smooth_kc6.28": dict(kc=2 * math.pi,     amplitude=6.0e-4, kpeak=8.0, seed=1, init_file=None),
}

here = os.path.dirname(os.path.abspath(__file__))
template = open(os.path.join(here, "template.in")).read()

for case, c in CASES.items():
    for theta in ANGLES:
        th = math.radians(theta)
        text = template
        f = c["init_file"]
        if callable(f):
            f = f(theta)
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
            "@RESTART_FILE@": f or "Dati.cart.out",
        }.items():
            text = text.replace(key, val)
        assert "@" not in text, "unreplaced placeholder"
        assert not (f and "ABSOLUTE/PATH" in f), "%s: init_file is still the placeholder path" % case
        d = os.path.join(here, "runs", case, "theta_%04.1f" % theta)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "hst.in"), "w") as f:
            f.write(text)
        print("wrote", os.path.relpath(os.path.join(d, "hst.in"), here))
