# Campaign decks, results and scripts

Everything needed to rerun the cases of
[`../../production_filter.md`](../../production_filter.md).

## Decks

One directory per case and shear direction, each with an `hst.in` for the
code of this branch (deck group `&production`):

| directory | case | angles theta [deg] | grid (nx, ny, nz) | S t |
|---|---|---|---|---|
| `box_3x2x1/theta_<deg>/` | original 3:2:1 box, no filter | 0, 7.5, 15, 22.5, 30, 37.5, 45 | 48, 95, 24 | 2040 |
| `filter_sharp_kc3.80/theta_<deg>/` | 15 x 2 x 5 box, sharp filter at k_c = 3.80 | 0, 15, 30, 45 | 240, 95, 120 | 1400 |
| `filter_smooth_kc3.63/theta_<deg>/` | 15 x 2 x 5 box, smooth filter at k_c = 2 pi/sqrt 3, edge 0.3 | 0, 15, 30, 45 | 240, 95, 120 | 1400 |
| `filter_sharp_kc3.63/theta_<deg>/` | 15 x 2 x 5 box, sharp filter at k_c = 2 pi/sqrt 3 (the fourth curve of the figures) | 0, 15, 30, 45 | 240, 95, 120 | 1400 |

The shear has magnitude 1 and direction theta: `S = cos theta`,
`s2_amplitude = sin theta` with `s2_period = 0` (constant).

## Running

    make GPU=1 GPU_ARCH=cc80            # or `make` for the CPU build (48 h per small-box run on one core)
    cd <a copy of the case directory> && mpirun -np 1 <repo>/build-gpu/hst hst.in > run.log

One rank and one GPU per run (the rank count must divide nx + 1 and the
dealiased z size).  On one A100: about 1 h for a small-box run (0.008
s/step), 4.5 h for a filtered one (0.079 s/step, 220 000 steps, a few GB
of GPU memory).  `tools/horeka_group.slurm` is the job script of the
campaign: four runs on one node, one per GPU; it sets `wall_max` from the
job limit and resumes a run that has a restart file.  A deck starts from
the random field unless the run directory holds `Dati.cart.out`
(`restart_file`); set `wall_max` and `time_from_restart = .true.` to
split a run over several jobs.

Each run writes `statistics/Runtimedata`,
`statistics/variances_runtime.dat`, with the filter
`statistics/production_runtime.dat`, the column spectra
`statistics/spectra/exz<i>.dat` and a velocity snapshot `fields/field<i>.fld`
every 200 time units.

## Results

`results/all_modes.csv` (shear-frame statistics of every run, with block
errors `err_*`; the small box also at theta > 45), `results/forced_range.csv`
(the same from the modes at or above the forcing shell),
`results/filter_diagnostics.csv` (unforced share of the energy and of the
dissipation, production/dissipation, the producing share of the stress,
anisotropy of the unforced (`bl_*`) and forced (`bh_*`) ranges).

## Scripts

`tools/` holds the scripts that produced the tables and figures, as they
were run.  They read the layout and the conventions of the campaign
binary (run-time files in the run directory, box-height integrals, the
old column order of `variances_runtime.dat`; see section 5 of the
document) and the directory names of the campaign (`n1` = `box_3x2x1`,
`n5_prod3.80` = `filter_sharp_kc3.80`, `n5_prod3.63_edge0.3` =
`filter_smooth_kc3.63`, `n5_prod3.63` = `filter_sharp_kc3.63`), so they
need adapting before they run on new output.

| script | output |
|---|---|
| `analyse_small_box.py`, `analyse.py` | shear-frame statistics and block errors (`results/all_modes.csv`) |
| `prod.py` | `results/filter_diagnostics.csv` |
| `highband.py` | `results/forced_range.csv`, `figures/angle_dependence_forced_range.png` |
| `plot_smooth.py` | `figures/angle_dependence_all_modes.png` |
| `compspec_extract.py`, `doc_figures.py` | per-column component spectra from the snapshots; the spectra and time-series figures and the band tables |
