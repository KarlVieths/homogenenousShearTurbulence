#!/usr/bin/env python3
"""Two checks for the oscillating-S2 Kelvin tests (companion of kelvin_ref.py).

order   Order-of-accuracy test (Option 1). Three runs of the same case at
        dt, dt/2, dt/4. Errors against the exact solution, evaluated at each
        run's own output times; observed order from the two finest runs.

            python3 kelvin_checks.py order run_dt.csv run_dt2.csv run_dt4.csv \
                --kx 0 --kz 1.2566370614 --ky0 3.1415926536 --G 1 \
                --S 1 --A 1 --T 1.75 --nu 0 --v0 1+0.3j

        The weights of the two mechanisms of the production can be set apart
        (--glu for the lift-up, --gorr for the Orr transfer; default G each) and
        the order taken in eta instead of v (--qty eta, with --eta0 the initial
        eta; the run files then need the columns Re_eta, Im_eta, which
        test_kelvin_s2's tables have).  The suite itself runs these order tests
        inside test_kelvin_s2 (decks kelvin_split_order_*, order_fine_min = 2.8).

        Exit status: 0 PASS, 1 FAIL, 2 INVALID (finest error at round-off,
        dt not halving, or files inconsistent).

margin  Seeded-bug audit (Option 2). Reads a long-format table of errors,
        one row per (bug, case, quantity), and prints the margin of each bug
        over the tolerance as a markdown table.

            python3 kelvin_checks.py margin bug_errors.csv [--robust 10]

        bug_errors.csv columns:  bug,case,quantity,error,tol
        Use bug = "none" for the correct solver; those rows give the headroom
        and a suggested per-case tolerance.

Run-file format (one file per run): CSV with a header and at least the
columns  t, Re_v, Im_v ; optional  dt  (time step of the run, constant).
Times are the run's actual output times; the reference is evaluated there,
so no interpolation of the numerical solution is needed.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np

from kelvin_ref import closed_form

P_MIN = 2.8          # required observed order on the finest pair (nu ~ 0, RK3)
E_FLOOR = 1e-11      # finest error must exceed this, else the order is meaningless


def round_up_1sig(x):
    """Round x > 0 up to one significant digit (1.2e-7 -> 2e-7)."""
    e = math.floor(math.log10(x))
    m = x / 10.0**e
    return math.ceil(m - 1e-12) * 10.0**e


def read_run(path):
    from kelvin_ref import read_csv     # skips the '#' parameter lines of test_kelvin_s2's tables
    _, data = read_csv(path)
    t = np.atleast_1d(data["t"])
    v = np.atleast_1d(data["Re_v"]) + 1j * np.atleast_1d(data["Im_v"])
    dt = float(np.atleast_1d(data["dt"])[0]) if "dt" in data else None
    order = np.argsort(t)
    eta = None
    if "Re_eta" in data:
        eta = (np.atleast_1d(data["Re_eta"]) + 1j * np.atleast_1d(data["Im_eta"]))[order]
    return t[order], v[order], dt, eta


def max_rel_error(t, v, p, eta=None):
    w = 2.0 * np.pi / p.T
    v_ref, eta_ref, _ = closed_form(t, p.kx, p.ky0, p.kz, p.G, p.S, p.A, w, p.nu, p.v0, p.eta0,
                                    g_lu=p.glu, g_orr=p.gorr)
    if p.qty == "eta":
        if eta is None:
            raise SystemExit("--qty eta needs the columns Re_eta, Im_eta in the run files")
        return float(np.max(np.abs(eta - eta_ref)) / np.max(np.abs(eta_ref)))
    return float(np.max(np.abs(v - v_ref)) / abs(p.v0))


def cmd_order(p):
    runs = [read_run(f) for f in p.runs]
    errs = [max_rel_error(t, v, p, eta) for (t, v, _, eta) in runs]
    dts = [dt for (_, _, dt, _) in runs]
    print("run                      dt            " + ("max |eta - eta_ref|/max|eta_ref|" if p.qty == "eta" else "max |v - v_ref|/|v0|"))
    for f, dt, e in zip(p.runs, dts, errs):
        print(f"{f:24s} {dt if dt is not None else float('nan'):<13.6e} {e:.3e}")

    if all(dt is not None for dt in dts):
        r1, r2 = dts[0] / dts[1], dts[1] / dts[2]
        if abs(r1 - 2.0) > 1e-12 * 2 or abs(r2 - 2.0) > 1e-12 * 2:
            print(f"INVALID: dt ratios {r1:.15g}, {r2:.15g} are not exactly 2")
            return 2
    if min(errs) <= 0.0:
        print("INVALID: zero error (identical to the reference?)")
        return 2

    p_coarse = math.log2(errs[0] / errs[1])
    p_fine = math.log2(errs[1] / errs[2])
    print(f"observed order: coarse pair {p_coarse:.2f}, fine pair {p_fine:.2f}")

    if errs[2] < E_FLOOR:
        print(f"INVALID: finest error {errs[2]:.1e} below {E_FLOOR:.0e}; use a larger dt")
        return 2
    if p_fine >= P_MIN:
        print(f"PASS: order {p_fine:.2f} >= {P_MIN}")
        return 0
    hint = ""
    if 0.8 <= p_fine <= 1.2:
        hint = " (looks first order: S2, gamma or the extra term at the wrong time)"
    elif 1.8 <= p_fine <= 2.2:
        hint = " (looks second order: one S2 per step, or nu not small enough)"
    print(f"FAIL: order {p_fine:.2f} < {P_MIN}{hint}")
    return 1


def cmd_margin(p):
    rows = list(csv.DictReader(open(p.table)))
    ok = defaultdict(dict)                 # case -> quantity -> (error, tol)
    bugs = defaultdict(list)               # bug -> [(ratio, case, quantity, error, tol)]
    for r in rows:
        e, tol = float(r["error"]), float(r["tol"])
        if r["bug"].strip().lower() == "none":
            ok[r["case"]][r["quantity"]] = (e, tol)
        else:
            bugs[r["bug"]].append((e / tol, r["case"], r["quantity"], e, tol))

    print("### Correct solver: headroom per case\n")
    print("| Case | Quantity | Error | Tolerance | Headroom tol/error | Suggested tolerance (3 x error) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for case in sorted(ok):
        for q, (e, tol) in sorted(ok[case].items()):
            sug = round_up_1sig(3.0 * e) if e > 0 else float("nan")
            print(f"| {case} | {q} | {e:.1e} | {tol:.0e} | {tol / e if e > 0 else float('inf'):.1f} | {sug:.0e} |")

    print("\n### Seeded bugs: margin at the suite's time step\n")
    print("| Bug | Detecting case | Quantity | Error | Tolerance | Margin | Verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    status = 0
    for bug in sorted(bugs):
        ratio, case, q, e, tol = max(bugs[bug])
        if ratio >= p.robust:
            verdict = "robust"
        elif ratio >= 1.0:
            verdict = "caught, fragile"
            status = max(status, 1)
        else:
            verdict = "missed by amplitude checks"
            status = max(status, 1)
        print(f"| {bug} | {case} | {q} | {e:.1e} | {tol:.0e} | {ratio:.1f} | {verdict} |")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("order", help="order-of-accuracy test from three runs")
    o.add_argument("runs", nargs=3, help="run files at dt, dt/2, dt/4")
    for name, default in (("kx", 0.0), ("kz", 2 * np.pi / 5.0), ("ky0", np.pi), ("G", 1.0),
                          ("S", 1.0), ("A", 1.0), ("T", 1.75), ("nu", 0.0)):
        o.add_argument(f"--{name}", type=float, default=default)
    o.add_argument("--v0", type=complex, default=1.0 + 0.3j)
    o.add_argument("--eta0", type=complex, default=0.0, help="initial eta (default 0: v is checked, eta does not enter it)")
    o.add_argument("--glu", type=float, default=None, help="weight of the lift-up (default G)")
    o.add_argument("--gorr", type=float, default=None, help="weight of the Orr transfer (default G)")
    o.add_argument("--qty", choices=("v", "eta"), default="v", help="quantity whose order is taken")

    m = sub.add_parser("margin", help="seeded-bug margin table")
    m.add_argument("table", help="CSV with columns bug,case,quantity,error,tol")
    m.add_argument("--robust", type=float, default=10.0, help="margin regarded as robust")

    p = ap.parse_args()
    sys.exit(cmd_order(p) if p.cmd == "order" else cmd_margin(p))


if __name__ == "__main__":
    main()
