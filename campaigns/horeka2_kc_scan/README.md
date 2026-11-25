# Campaign: production filter at k_c = 2 pi/3 and 2 pi, four shear directions, HoreKA2

8 runs, each on one GPU: the smooth production filter (edge 0.3) at two
cutoffs, shear direction theta = 0, 15, 30, 45 deg.  Everything else is
fixed and is that of the campaign of `doc.production/production_filter.md`
(`filter_smooth_kc3.63`): box 15 x 2 x 5, grid 240 x 95 x 120, Re = 1000,
S t = 1400, random initial field (amplitude 6e-4, kpeak 8, seed 1).

| case | k_c | forced where |
|---|---|---|
| `runs/smooth_kc2.09/` | 2 pi/3 = 2.094 | k_h >= 0.7 k_c ... 1.3 k_c (G from 0 to 1) |
| `runs/smooth_kc6.28/` | 2 pi   = 6.283 | the same |

`runs/<case>/theta_<deg>/hst.in` are the decks, written by `make_runs.py`
from `template.in` (only `S = cos theta`, `s2_amplitude = sin theta`,
`prod_kc` and the header differ from the 3.63 deck of the earlier campaign).

## Files

| file | what |
|---|---|
| `template.in` | the deck with placeholders |
| `make_runs.py` | writes `runs/*/theta_*/hst.in`; the cases, angles, edge and initial condition (`CASES`) are at its top |
| `horeka2_group.slurm` | the job: four runs on one node, one per GPU (details in its header) |
| `submit.sh` | submits one job per case, or a short test |
| `status.sh` | state, time reached and s/step of every run |

## Steps

1. **Build** on a login node, in the repository (this is `compile-horeka2.sh`
   at the repository root; it has not run successfully on HoreKA2 yet, see
   `env/horeka2.sh`):

        cd ~/hst && ./compile-horeka2.sh      # one build job per GPU type, on the matching nodes; --status shows what is built

   The job picks `build-gpu-<device>/hst` (h100, h200 or b200) from the GPU's name;
   an H100 or H200 node without a build of its own uses `build-gpu-h100`.

2. **Copy this directory to a workspace** (a few GB per run; allocate one with
   `ws_allocate kc_scan 60`, find it with `ws_find kc_scan`; no backup; the
   documentation gives the pattern `/hfs2/work/workspace/scratch/<user>-<name>`) and work there:

        cp -r ~/hst/campaigns/horeka2_kc_scan $(ws_find kc_scan)/ && cd $(ws_find kc_scan)/horeka2_kc_scan

3. **Test** (8 minutes, one case, four angles at once):

        export HST_ACCOUNT=<project account>  HST_PARTITION=<GPU partition>
        export HST_TEST_PARTITION=<development partition>    # optional
        ./submit.sh test

   Afterwards `./status.sh test` and `grep s/step test/*/theta_*/run-*.log | tail`
   give the speed (A100: 0.079 s/step for this box; the whole run is 220 000
   steps).  Check that the four logs show four different GPUs working
   (`nvidia-smi` in the job log) and no MPI errors.  If `mpirun -np 1` does
   not start inside the allocation, submit with `LAUNCH= ./submit.sh test`
   (the binary then runs directly, as one MPI process).

4. **Run**:

        export TIME=<limit>        # default 06:00:00; at most what the partition allows
        ./submit.sh

   Two jobs, one per case.  A run that the time limit stops writes
   `Dati.cart.out` (restart every 20 time units and at the end) and is
   marked `INCOMPLETE`; the job then submits a follow-up job for those runs,
   which resume from it (up to 10 jobs in a chain, `MAXSEG`).  A run that
   ends otherwise is marked `FAILED` and is not resubmitted.  `./status.sh`
   shows the state at any time; `scancel` the pending job to stop a chain.

5. **Results** are in each run directory: `statistics/Runtimedata`,
   `statistics/variances_runtime.dat`, `statistics/production_runtime.dat`,
   column spectra `statistics/spectra/exz<i>.dat` and snapshots
   `fields/field<i>.fld` every 200 time units (columns of the files:
   `src/hst_stats.f90`).  The analysis scripts of the earlier campaign are in
   `doc.production/assets/campaign/tools/` and need adapting to the layout
   (see the README there).

## Things to know

* **Initial field** (`&init` and `restart_file` of the deck, set per case in
  `CASES` of `make_runs.py`, then rerun it).  Two ways:
  1. *Random field* (the default, `init_file=None`): the code generates it when
     `restart_file` (default `Dati.cart.out`) does not exist, from `amplitude`,
     `kpeak` and `seed` of `&init`: a random divergence-free field with the
     spectrum (k/kpeak)^2 exp(1 - (k/kpeak)^2), no mean profiles.  `amplitude`
     scales the potential, not the rms, and decides whether the run becomes
     turbulent (`FINDINGS.md`: 1e-3 at kpeak 4 decays at Re = 1000).  For a
     laminar-looking start, raise `amplitude`, or change `seed`.
  2. *A turbulent field from a file* (`init_file=<path>` or a function of
     theta, e.g. `from_3_63`: the end of the `filter_smooth_kc3.63` run of the
     same angle, a `Dati.cart.out` or a `fields/field<i>.fld` snapshot).  The
     deck then has `restart_file = '<path>'` and `time_from_restart = .false.`,
     so the clock starts at `time = 0` and `t_max = 1400` is 1400 time units
     after the start.  The file must have the same `alfa0`, `beta0` (box
     15 x 5; otherwise the run stops unless `restart_remap = 'mode_index'`, which
     keeps the modes the two boxes share); a different grid is allowed
     (modes copied, y resampled by nearest neighbour, which for the same
     grid changes nothing).  Re and S may differ (the log notes it).  The
     file must not be named `Dati.cart.out` in the run directory, and the
     path should be absolute: the code reads it relative to the run
     directory.  When a job resumes a run the script sets
     `restart_file = 'Dati.cart.out'` and `time_from_restart = .true.`
     itself, so the start file is read once.
  A start from the field of another filter has a transient (the first few
  hundred time units) that the analysis should discard.
* **Resubmitting a finished run is safe**: runs marked `DONE` are skipped.
  To rerun one from scratch remove its run directory contents except `hst.in`.
* **The restart files and statistics are appended**, never truncated; after a
  crash that left a `Dati.cart.out` the next job continues from it and
  `Runtimedata` then has the lines of the lost interval twice (about 20 time
  units at most), which the analysis should drop.
* **A node whose GPUs do not initialise** (nvidia-smi lists them, CUDA returns error 999) makes the job stop at its pre-flight check with a message.  Keep such a node out of the next submission with `export SBATCH_EXCLUDE=<node>` (Slurm reads it as `--exclude`) before `./submit.sh`.
* **Not verified on HoreKA2**: module names and MPI of `env/horeka2.sh`, the
  Slurm account/partition names, `--gres=gpu:4`, `--cpus-per-task=8` (change
  the header of `horeka2_group.slurm` if the node does not have 4 x 8 cores
  to give), and the time limit of the partitions.  `HST_GPUS=gpu:h100:4`-style
  GRES strings can be given to `submit.sh` if the cluster needs them (the count
  after the last colon is what it uses).
* **Partitions of HoreKA2** (binary = `build-gpu-<device>` of the repository; nodes,
  limits and status from docs.nhr.kit.edu/clusters/horeka-2 and /usage/slurm, read 2026-11-19):

  | partition | GPUs per node | nodes | binary | `HST_GPUS` | status |
  |---|---|---|---|---|---|
  | `gpu-h100` (dev: `dev-gpu-h100`, 1 h, 1 node) | 4 H100 | 22 | `build-gpu-h100` | `gpu:4` (default) | operational |
  | `gpu-h200` (no dev partition) | 4 H200 | 13 | `build-gpu-h200` | `gpu:4` (default) | operational |
  | `gpu-h200-8` | 8 H200 | 1 | `build-gpu-h200` | `gpu:8`: all 8 runs in one job | operational, one node only |
  | `gpu-b200` (dev: `dev-gpu-b200`) | 4 B200 | 75 | `build-gpu-b200` | `gpu:4` | "Operation will start in December 2026" |

  The time limit of the GPU partitions is 2 days, of the dev partitions 1 hour,
  and a dev partition allows one job per user at a time.  `sinfo_t_idle` lists
  the idle nodes only (a plain `sinfo` is restricted), so 0 there means busy,
  not absent.
  With `gpu:4` `submit.sh` submits one job per case; with `gpu:8` one job with
  all eight runs (and the test runs both cases at once).
