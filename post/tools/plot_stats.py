"""Plot the run statistics from ``Runtimedata`` (energy, dissipation, stresses).

Usage::

    python3 -m post plot_stats [deck] [-o out.png]

Reads ``Runtimedata`` next to the deck and plots energy, dissipation and the
Reynolds stresses against time.  Requires ``matplotlib`` (imported lazily).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..io import read_runtimedata, read_variances


def run(cfg, argv) -> int:
    p = argparse.ArgumentParser(prog="plot_stats",
                                description="Plot Runtimedata (energy, diss, stresses).")
    p.add_argument("-o", "--out", default=None, help="output image path")
    p.add_argument("--log", action="store_true", help="logarithmic y for energy/diss")
    p.add_argument("--no-variances", action="store_true",
                   help="only Runtimedata, skip variances_runtime.dat")
    args = p.parse_args(argv)

    rt_file = cfg.run_dir / "Runtimedata"
    var_file = cfg.run_dir / "variances_runtime.dat"

    rt = read_runtimedata(rt_file)
    t = rt["time"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(t, rt["energy"], label="E = <u_i u_i>/2 (ly/2 <..>)")
    ax.plot(t, rt["diss"], label="diss = <∂_j u_i ∂_j u_i>/2")
    if args.log:
        ax.set_yscale("log")
    ax.set_xlabel("time"); ax.set_ylabel("E, diss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, rt["uw2"], label="u v/2")
    ax.plot(t, rt["vw2"], label="w v/2")
    ax.set_xlabel("time"); ax.set_ylabel("stress"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, rt["deltat"], label="deltat")
    ax.plot(t, rt["cfl_dt"], label="cfl*deltat")
    ax.set_xlabel("time"); ax.set_ylabel("Δt, cfl·Δt"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if var_file.exists() and not args.no_variances:
        var = read_variances(var_file)
        ax.plot(var["time"], var["uu"], label="uu")
        ax.plot(var["time"], var["vv"], label="vv")
        ax.plot(var["time"], var["ww"], label="ww")
        ax.set_xlabel("time"); ax.set_ylabel("variances (ly <..>)")
    else:
        ax.plot(t, rt["meanflowx"], label="meanflowx")
        ax.plot(t, rt["meanflowy"], label="meanflowy")
        ax.set_xlabel("time"); ax.set_ylabel("mean flow integral")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"{rt_file.parent}  (Re={cfg.re}, S={cfg.S})")
    fig.tight_layout()
    out = args.out or (cfg.run_dir / "statistics.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0
