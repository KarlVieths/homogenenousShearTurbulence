# Seed ensemble at theta = 30 deg, square box, HoreKA2

Four runs of one case that differ only in the seed of the random initial field:
the scatter of their time-averaged statistics, compared with the quoted block
errors, calibrates the error bars (and gives the noise floor for comparing two
single runs).  Box 10.5 x 2 sqrt 3 x 10.5, smooth production filter (edge 0.3) at
k_c = 2 pi/sqrt 3 (l_f = sqrt 3), shear direction 30 deg (`S = cos`, `s2_amplitude = sin`,
constant), Re = 1000, grid 168 x 165 x 252, S t = 1400.  Everything but `seed` is
fixed (`diff` of two decks: the seed line).

| run | directory | seed |
|---|---|---|
| R1 | `runs/square_th30_s101/` | 101 |
| R2 | `runs/square_th30_s102/` | 102 |
| R3 | `runs/square_th30_s103/` | 103 |
| R4 | `runs/square_th30_s104/` | 104 |

The default deck uses `seed = 1`; any other distinct integers do.  The in-progress
square-box run at 30 deg is a fifth sample if its settings are identical
(`analyse_seeds.py --extra <its run directory>`).

## Files

| file | what |
|---|---|
| `template.in` | the deck with placeholders (the 15 x 2 x 5 campaign's deck with the square box) |
| `make_runs.py` | writes `runs/*/hst.in`; the seeds, angle, filter and `AMPLITUDE` are at its top |
| `horeka2_group.slurm` | the job: runs on one node, one per GPU (the script of `campaigns/horeka2_kc_scan`; header describes resume and chaining) |
| `submit.sh` | `./submit.sh test` (8 minutes) or `./submit.sh` (the real job) |
| `check.sh` | `./check.sh decks` (only the seed differs), `./check.sh logs [test]` (seeds, first q2, s/step) |
| `status.sh` | state, time reached, s/step of every run |
| `analyse_seeds.py` | the analysis below |

## Steps

1. **Build** in the repository: `./compile-horeka2.sh` (it submits one build job per GPU type to the matching partition and waits; `--status` shows what is built); the job picks
   `build-gpu-<device>/hst` from the GPU's name (the build of the 15 x 2 x 5 campaign).
2. **Copy** this directory to a workspace (about 10-13 GB per run; `ws_find` gives the path of an allocated one) and work there:

        cp -r <repo>/campaigns/square_seed_ensemble $(ws_find <name>)/ && cd $(ws_find <name>)/square_seed_ensemble

3. **Check the decks**: `./check.sh decks`.
4. **Test** (four runs, one per GPU, 8 minutes: the start, the CFL step, seconds per step for the time limit):

        export HST_ROOT=<repository>  HST_ACCOUNT=<account>  HST_PARTITION=gpu-h100
        export HST_TEST_PARTITION=dev-gpu-h100       # optional, 1 h limit
        ./submit.sh test
        ./check.sh logs test ; ./status.sh test ; grep -h s/step test/*/run-*.log | tail -4

   `check.sh logs test` shows four distinct seeds and a first q2 of about 0.11 in all four.
5. **Run**: `export TIME=12:00:00` (the default; at most the partition's 2 days), then `./submit.sh`.
   One job on one node with the four runs.  A run that the limit stops writes `Dati.cart.out`, is marked
   `INCOMPLETE` and the job submits a follow-up (up to 10 jobs; `scancel` the pending one to stop the chain).
   `./status.sh` at any time.
6. **Analyse** (needs numpy only):

        python3 analyse_seeds.py [--transient 400] [--block 50] [--extra DIR]

## Cost

Estimate, to be replaced by the timing job: 0.052 s/step on an H100 for the 15 x 2 x 5 grid (240 x 95 x 120)
with four runs per node, times 2.55 more points = 0.13 s/step, times about 270 000 steps (the dt of the
15 x 2 x 5 test, CFL 0.7) = about 10 h per run, 40 GPU-hours for the set (the document's A100 estimate:
8-12 h).  Disk: about 0.7 GB per snapshot, 14 snapshots (`dt_field = 100`) and the restart file.

## Initial field: `amplitude` is not the document's 0.003

`amplitude` scales the random potential per mode, so q2 at t = 0 grows with the number of modes (q2 ~
amplitude^2 x modes).  0.003 is the small box's value (q2 = 0.12 there).  For this box it gives a first q2
of **2.1**, not about 0.1 (a run of the code's initial field, 2 steps on the CPU).  `AMPLITUDE = 7e-4` in
`make_runs.py` gives **0.113** (0.1132-0.1137 for the four seeds).  The in-progress run's value was not
available when this was written: use its amplitude instead if you want the same start, and rerun `make_runs.py`.

## Analysis (`analyse_seeds.py`)

Reads `statistics/Runtimedata`, `variances_runtime.dat`, `production_runtime.dat` of every run (the
conventions of `doc.production/assets/campaign/tools/analyse_small_box.py`; restart overlaps removed).

* Per run, `seed_runs.csv`: means for t >= 400 with the standard error from 50-time-unit blocks, for q2, eps,
  S*, -<u_par v>/q2, L_eps/l_f, b_par, b_vv, b_perp, <u_perp v>/q2, rho_par,perp and the unforced share of q2
  (the columns with G < 1); the integral autocorrelation time of q2(t); the means from t >= 600.
* Checks of the error bars, `seed_blocks.csv`: the standard error for blocks of 25, 50, 100, 200 (it should
  plateau), the means from 400 against 600.
* Across the runs, `seed_ensemble.csv`: mean, seed-to-seed standard deviation, T = sum ((x_i - mean)/SE_i)^2 ~
  chi^2(N-1) with its p-value, R = sqrt(T/(N-1)) (the factor to inflate the quoted errors), and 2.77 sd
  (two single runs of different configurations differ significantly at about 95 % only if they differ by more
  than that).  N = 4 gives the sd to about 40 %, so R is a rough calibration; compare it between the quantities.

The script ran on synthetic data (correlated noise) and its chi^2 tail agrees with the standard table values
(p = 0.05 at T = 7.81 for 3 degrees of freedom); it has not run on output of the code.

## Where this differs from the planning document

The values the document marks as to be confirmed against the in-progress deck were not available (that
deck is not in the repository); the decks follow the 15 x 2 x 5 campaign's deck (`filter_smooth_kc3.63`) where the document's
template left a choice:

* `cflmax = 0.7`, as in the campaign (the document had 1.0); `dt_stat = 0.1` from the time scales of the flow (integral time 7, bursting 15-20; see `campaigns/square_robustness/README.md`, "Sampling intervals"; the campaign had 0.01 and the document 0.5); `dt_field = dt_save = 100`
  as in the document, so a crash loses up to 100 time units (about 1 h);
* `time_from_restart = .false.` in the deck (the job sets it when it resumes; with `.true.` and no
  restart file the code starts at 0 anyway);
* `amplitude`, see above; the rest of the namelist (`exact_shift`, `ystretch = 0`, ...) is the campaign's.

To confirm against your square-box deck: `nx, ny, nz` (the values are scaled per unit length from 15 x 2 x 5),
`dt_stat`, `cflmax`, the averaging start (t = 400) and `amplitude`/`kpeak`; the job script (4 runs per node, one
per GPU, 12 h segments) replaces `jobs/horeka_prod.slurm` of the document, which is the HoreKA (A100) job.

## Things to know

* **Empty run directories.** A leftover `Dati.cart.out` is read instead of the random field; `submit.sh`
  refuses to submit a run directory that has one without a run log.
* **A node whose GPUs do not initialise** stops the job at its pre-flight check; keep it out with
  `export SBATCH_EXCLUDE=<node>`.
* **Not verified**: the cluster side is that of `campaigns/horeka2_kc_scan` (the test of 2026-11-19 on gpu-h100 worked); the
  square-box speed and memory are estimates until the test job has run.
* HoreKA2 runs no jobs between 2026-11-20 (noon) and about 4-7 December (shutdown).
