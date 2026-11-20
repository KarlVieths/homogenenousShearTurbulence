"""Ensemble mean of switch-off runs on the time after the switch (selective_lu_orr.md, section 7.1).

Usage::

    python3 -m post.python ensemble RUN1 [RUN2 ...] [--tau-max 300] [--dtau 1] [-o ensemble.dat]
                                    [--t-switch T1 [T2 ...]]
    python3 -m post.python ensemble --selftest

Every RUN is a run directory with ``statistics/Runtimedata``, ``variances_runtime.dat`` and
``split_runtime.dat``, made by a restart from a saved field of the reference run with only the
``&production`` group changed.  Each run is aligned on tau = t - t_switch (t_switch: the first time
of its files, or ``--t-switch``, one value per run, when the files hold the history before), put on
the common grid tau = 0, dtau, ..., tau_max by linear interpolation, and averaged over the runs.
The table written (``-o``, default stdout; columns ``tau`` and, for each quantity, ``<name>`` and
``<name>_se``, the mean and its standard error) holds

    q2 = 2 energy, diss, uv, vw                      (Runtimedata)
    b_uu b_vv b_ww b_uv b_vw b_uw                    (Reynolds stress anisotropy <u_i u_j>/q2 - delta_ij/3)
    E_tor E_pol P_LU P_Orr gP_LU gP_Orr              (split_runtime.dat)

The runs with a time axis that does not reach tau_max are dropped from that grid point on
(a decayed run ended early: it counts with its last value, i.e. it is held, ``--hold``, or the
ensemble is cut where the first run ends, the default).  The standard error needs two runs.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..io import find_run_file, read_runtimedata, read_split_runtime, read_variances

QUANTITIES = ("q2", "diss", "uv", "vw", "b_uu", "b_vv", "b_ww", "b_uv", "b_vw", "b_uw",
              "E_tor", "E_pol", "P_LU", "P_Orr", "gP_LU", "gP_Orr")


def load_run(run_dir, t_switch=None):
    """The quantities of one run on its own time axis: (tau, array (n, len(QUANTITIES)))."""
    rt = read_runtimedata(find_run_file(run_dir, "Runtimedata"))
    var = read_variances(find_run_file(run_dir, "variances_runtime.dat"))
    sp = read_split_runtime(find_run_file(run_dir, "split_runtime.dat"))
    if not (len(rt) == len(var) == len(sp)) or np.max(np.abs(rt["time"] - sp["time"])) > 1e-9:
        raise ValueError(f"{run_dir}: Runtimedata, variances_runtime.dat and split_runtime.dat do not have the same rows "
                         f"(a restart appended to older files?)")
    t = rt["time"]
    q2 = var["uu"] + var["vv"] + var["ww"]
    cols = {"q2": q2, "diss": rt["diss"], "uv": rt["uv"], "vw": rt["vw"],
            "b_uu": var["uu"] / q2 - 1 / 3, "b_vv": var["vv"] / q2 - 1 / 3, "b_ww": var["ww"] / q2 - 1 / 3,
            "b_uv": rt["uv"] / q2, "b_vw": rt["vw"] / q2, "b_uw": var["uw"] / q2}
    for name in ("E_tor", "E_pol", "P_LU", "P_Orr", "gP_LU", "gP_Orr"):
        cols[name] = sp[name]
    t0 = t[0] if t_switch is None else t_switch
    return t - t0, np.column_stack([cols[n] for n in QUANTITIES])


def ensemble(runs, tau_max, dtau, hold=False):
    """Ensemble mean and standard error of the quantities of ``runs`` (a list of (tau, array)) on the grid
    tau = 0, dtau, ..., tau_max.  Returns (tau, mean, se, n): n the number of runs at each grid point."""
    tau = np.arange(0.0, tau_max + dtau / 2, dtau)
    vals = np.full((len(runs), tau.size, len(QUANTITIES)), np.nan)
    for i, (t, a) in enumerate(runs):
        ok = tau <= t[-1] + 1e-9 if not hold else np.ones(tau.size, bool)
        for c in range(a.shape[1]):
            vals[i, ok, c] = np.interp(tau[ok], t, a[:, c])      # np.interp holds the last value beyond t[-1]
    n = np.sum(~np.isnan(vals[:, :, 0]), axis=0)
    mean = np.nanmean(vals, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nanstd(vals, axis=0, ddof=1) / np.sqrt(n)[:, None]
    se[n < 2] = np.nan
    return tau, mean, se, n


def _table(tau, mean, se, n):
    head = "tau n " + " ".join(f"{q} {q}_se" for q in QUANTITIES)
    body = [np.column_stack([tau, n] + [c for q in range(len(QUANTITIES)) for c in (mean[:, q], se[:, q])])]
    return head, body[0]


def selftest() -> int:
    """Three synthetic runs with a known mean (q2 = a exp(-r tau), a = 1, 2, 3 at r = 0.1) on shifted clocks."""
    rng_t = np.arange(0.0, 50.0 + 1e-9, 0.5)
    with tempfile.TemporaryDirectory() as tmp:
        runs = []
        for k, (a, t0) in enumerate(((1.0, 7.0), (2.0, 100.0), (3.0, 0.0))):
            d = Path(tmp) / f"run{k}" / "statistics"
            d.mkdir(parents=True)
            t = t0 + rng_t
            q2 = a * np.exp(-0.1 * rng_t)
            uu, vv, ww, uw = 0.5 * q2, 0.2 * q2, 0.3 * q2, -0.1 * q2
            uv = -0.15 * q2
            rt = np.column_stack([t, 0 * t, 0 * t, 1 + 0 * t, 0 * t, 0 * t, 0 * t, 0.01 + 0 * t, 0 * t,
                                  0.5 * q2, 0.05 * q2, uv, 0.0 * t])
            np.savetxt(d / "Runtimedata", rt)
            np.savetxt(d / "variances_runtime.dat", np.column_stack([t, uu, vv, ww, uw]))
            sp = np.column_stack([t, 0.4 * q2, 0.1 * q2, -uv, 0 * t, -uv, 0 * t, 0.05 * q2, 0 * t, 0 * t, 0 * t])
            with open(d / "split_runtime.dat", "w") as f:
                f.write("# time E_tor ...\n")
                np.savetxt(f, sp)
            runs.append(load_run(Path(tmp) / f"run{k}"))
        tau, mean, se, n = ensemble(runs, 40.0, 2.0)
        q2m, q2se = mean[:, QUANTITIES.index("q2")], se[:, QUANTITIES.index("q2")]
        exp_mean = 2.0 * np.exp(-0.1 * tau)                     # (1 + 2 + 3)/3
        exp_se = np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3) * np.exp(-0.1 * tau)
        ok = (np.max(np.abs(q2m - exp_mean)) < 1e-12 and np.max(np.abs(q2se - exp_se)) < 1e-12 and np.all(n == 3)
              and abs(mean[0, QUANTITIES.index("b_uu")] - (0.5 - 1 / 3)) < 1e-12)
        # the shortest run ends at tau = 20: from there the others only, unless held
        short = [(t[t <= 20.0], a[t <= 20.0]) if i == 0 else (t, a) for i, (t, a) in enumerate(runs)]
        tau2, m2, s2, n2 = ensemble(short, 40.0, 2.0)
        ok &= bool(np.all(n2[tau2 <= 20.0] == 3) and np.all(n2[tau2 > 20.0] == 2))
        tau3, m3, s3, n3 = ensemble(short, 40.0, 2.0, hold=True)
        ok &= bool(np.all(n3 == 3))
    print("ensemble selftest:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def run(cfg, argv) -> int:
    p = argparse.ArgumentParser(prog="ensemble", description="Ensemble mean of switch-off runs on tau = t - t_switch.")
    p.add_argument("runs", nargs="*", help="further run directories (the first is the deck / run directory of the call)")
    p.add_argument("--tau-max", type=float, default=300.0)
    p.add_argument("--dtau", type=float, default=1.0)
    p.add_argument("--t-switch", type=float, nargs="*", default=None, help="t_switch of each run (default: the first time of its files)")
    p.add_argument("--hold", action="store_true", help="a run that ended early counts with its last value")
    p.add_argument("-o", "--out", default=None, help="output table (default: stdout)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest()
    dirs = [cfg.run_dir] + [Path(r) for r in args.runs]
    if args.t_switch is not None and len(args.t_switch) != len(dirs):
        p.error("--t-switch needs one value per run")
    runs = [load_run(d, None if args.t_switch is None else args.t_switch[i]) for i, d in enumerate(dirs)]
    tau, mean, se, n = ensemble(runs, args.tau_max, args.dtau, hold=args.hold)
    head, body = _table(tau, mean, se, n)
    fmt = ["%.6g", "%d"] + ["%.8e"] * (2 * len(QUANTITIES))
    if args.out:
        np.savetxt(args.out, body, fmt=fmt, header=head)
        print(f"{len(dirs)} runs, tau 0..{tau[-1]:g}: {args.out}")
    else:
        np.savetxt(sys.stdout, body, fmt=fmt, header=head)
    return 0
