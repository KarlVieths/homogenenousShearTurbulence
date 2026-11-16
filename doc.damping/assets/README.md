# Damping campaign: decks, figures and result tables

Material of the campaign that [`../DAMPING.md`](../DAMPING.md) section 4
reports (there it is cited by its place in the campaign directory, e.g.
`angle_dependence.png`, `tools/analyse.py`).

## Decks

`campaign/<case>/theta_<deg>/hst.in`, one per run, for the code of this
repository (deck group `&damping`):

| directory | case | angles theta [deg] | grid (nx, ny, nz) | S t |
|---|---|---|---|---|
| `n1/` | the 3:2:1 box, no damping | 0, 45 | 48, 95, 24 | 2040 |
| `n2_hard/` | box 6 x 2 x 2, hard mask | 0, 15, 30, 45 | 96, 95, 48 | 1400 |
| `n3_hard/` | box 9 x 2 x 3, hard mask | 0, 15, 30, 45 | 144, 95, 72 | 1400 |
| `n4_hard/` | box 12 x 2 x 4, hard mask | 0, 15, 30, 45 | 192, 95, 96 | 1400 |
| `n5_hard/` | box 15 x 2 x 5, hard mask | 0, 15, 30, 45 | 240, 95, 120 | 1400 |
| `n10_hard/` | box 30 x 2 x 10, hard mask | 0, 45 | 479, 95, 240 | 1400 |
| `n3_rate1/`, `n3_rate3/`, `n3_rate10/` | box 9 x 2 x 3, soft damping at rate 1, 3, 10 | 0, 15, 30, 45 | 144, 95, 72 | 1400 |

The decks are those of the runs, with two keys reset that the job script
of the campaign had set per job: `wall_max = 0` and
`time_from_restart = .false.` (a run from the random field in one piece;
set both to split a run over several jobs).  The shear has magnitude 1
and direction theta: `S = cos theta`, `s2_amplitude = sin theta` with
`s2_period = 0`.  Run as

    cd <a copy of the case directory> && mpirun -np 1 <repo>/build-gpu/hst hst.in > run.log

with one rank and one GPU per run up to n = 5; the n = 10 runs used four
ranks on four GPUs.  The runs of the campaign were made before the
run-time statistics changed (note of 2026-11-09 in `DAMPING.md` section
4); a rerun writes `statistics/Runtimedata` and
`statistics/variances_runtime.dat` with the present conventions.

## Figures

`figures/`, redrawn on 2026-11-16 from the runs listed above, with the
scripts of the campaign:

| file | content |
|---|---|
| `angle_dependence.png` | shear-frame statistics against the shear direction, all cases |
| `angle_dependence_hard.png` | the same, hard-mask cases and the small box only |
| `history.png` | running means of q2 and S* in time, by case |
| `spectra.png` | column energies E(kx, kz) in the shear frame with the damping ellipse, every run |
| `soft.png` | what the soft damping removes, against its rate |

## Result tables

`results/results.csv` (the statistics of `angle_dependence.png` with block
errors `err_*`; `n1` at 13 angles from 0 to 90) and `results/soft.csv`
(the numbers of `soft.png`).
