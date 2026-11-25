# Robustness campaign: the production filter in HST (stages 1a to 3b), HoreKA2

Does the filtered flow depend only on Re_c and e once the box is large enough?  30 runs in eight stages (0.45 TB of
output), each changing one thing from the baseline: the square box (A_h = 6.06, A_yc = 2, e = 0.3, Re_c = 3000; the box
of `campaigns/square_seed_ensemble`).  Plan, decks, criteria and the checklist are those of the planning document
"Robustness campaign - production filter in HST" (24 Nov 2026).

**Cost.**  The document counts A100 GPU-hours (about 690 for these 30 runs, 735 with stage 0).  HoreKA2 has no A100: its GPUs are H100, H200
and B200.  With the H100 speed measured on HoreKA2 the study (stage 0 + stages 1a-3b) is about **425 GPU-hours on H100** (658 in the conservative reading), 326 (506) on H200 and 212 (328) on B200
if each run is its own job; if jobs hold whole nodes (four runs per node) the held time is larger, 146 node-hours on H100 (226) ... 73 on B200 (113).  The H200 and B200 figures are estimates, not measurements, and about 5 % higher with the merged code at `dt_stat = 0.01` ("After the device-memory fix").
**Which unit is billed is open**: the documentation says GPU hours, you were told node-hours; "Billing" below says how to find out in two minutes.
See "Cost on HoreKA2" below; the A100 column of the stage table is the document's.

**Two things from the documentation (read 25 Nov 2026) change the plan, details at the end:** the B200 nodes are ARM (Grace) machines, so their binary must be built on a B200 node
(`./compile-horeka2.sh` now builds each binary on the right kind of node), and HoreKA2 is shut down until about 4-7 December.

**To run it, go to "How to run it: three steps"** (build and checks; stage 0; the rest).

**Stage 0 is not here**: it is the seed ensemble, `campaigns/square_seed_ensemble/` (its four runs are the document's
`square_th30_s101` to `s104`; checked: mesh 168 x 165 x 252, L = 10.5 x 2 sqrt 3 x 10.5, re = 1000, S = cos 30 deg, s2 = sin 30 deg, e = 0.3,
seeds 101-104 as in the document's manifest).  It runs first and independently.  Its result (`seed_ensemble.csv`: sigma_seed and R) is the
input of the analysis here.  The baseline runs at 0, 15, 30 and 45 deg are those of the in-progress square-box campaign and are not rerun.

| stage | purpose | varies | runs | GPU-h (A100, document) | starts after |
|---|---|---|---|---|---|
| 1a | convergence in A_h | A_h = 3, 4.5, 9 (theta 0, 30) | 6 | 70 | stage 0 |
| 1b | convergence in A_yc | A_yc = 1.5, 3 | 4 | 52 | 0 |
| 1c | A_xc != A_zc | 9 x 2 x 6 | 2 | 34 | 0 |
| 1d | remaining angles | theta = 22.5, 37.5 at the adopted box | 2 | 23 | 1a-c |
| 1e | resolution | 1.5 x grid, theta = 30 | 1 | 58 | 0 |
| 2 | edge | e = 0.15, 0.5 at A_h = 6.06 and 4.5 | 8 | 72 | 1a-c |
| 3a | laminar threshold | Re_c = 1000, 1500, 2000, theta = 0 | 3 | <= 35 | 1a-c |
| 3b | Re_c dependence | Re_c = 6000 at A_h = 6.06 (4 GPUs) and 4.5 | 4 | 346 | 1a-c, NCCL check |

## Files

| file | what |
|---|---|
| `manifest.csv` | one row per run (the document's columns, plus `est_mem_gb`, `est_disk_gb`); `est_wall_h` is in the document's A100 hours (divide by the speed factor for HoreKA2); written by `make_manifest.py` |
| `manifest_document.csv` | the document's manifest, verbatim (stages 1a-3b), the reference of `make_manifest.py --check` |
| `make_manifest.py` | the manifest from Re_c, the aspect ratios, e, theta (formulas of the document); `--adopt` another box for stages 1d, 2, 3; `--replicates` more seeds |
| `template.in`, `make_runs.py` | the deck template and the generator: `runs/<id>/hst.in` (an empty directory per run) |
| `check.py` | the pre-launch check of the decks against the manifest, the seeds and the seed ensemble's deck |
| `horeka2_runs.slurm` | the job: runs on one node, each on RUN_GPUS GPUs; resume, chaining, STOP file; refuses a binary of the wrong CPU architecture (header) |
| `submit.py` | the plan and the submission: groups the runs into jobs, time limits, gates, `--test`, `--dry-run` |
| `status.py` | state, t, s/step, q2 and the laminar check of every run |
| `analyse.py`, `hstlib.py` | the per-run statistics and the comparisons of the document (below) |

## How to run it: three steps

| step | what | wall time | done when |
|---|---|---|---|
| 1. Build and initial checks | env and builds, both campaign folders on the workspace, offline checks, the 8-minute tests (stage 0 and the timing pilot), the billing and chain checks | 1-2 h plus queue | stage-0 test shows four distinct seeds and a first q2 of about 0.11; the pilot gives a speed factor; the billing unit is known and a chain works |
| 2. Stage 0 | the four seed runs, then their analysis | 8-11 h on an H100 | `seed_ensemble.csv` read; the number of seeds per point is decided |
| 3. The rest | 3A: stages 1a, 1b, 1c, 1e; review; 3B: stages 1d, 2, 3a, 3b | days (cost: "Cost on HoreKA2") | `analyse.py` run on everything |

Placeholders in `<...>` are yours.  Everything runs on a login node except the jobs.  Commands are meant to be pasted one block at a time.

### Step 1: build and initial checks

**1.1 Variables** (put them in a file and `source` it in every new shell):

```
export HST_ROOT=/home/ka_istm/ka_th5982/hst-tools/hst
export HST_ACCOUNT=hk-project-p0024333
export WS=$(ws_find hst-1)         # /hfs2/work/workspace/scratch/ka_th5982-hst-1
```

**1.2 Space, queue, data**

```
ws_list                            # lifetime; at most 3 extensions of 60 days: ws_extend hst-1 60 (expiry is suspended during the shutdown)
/usr/lpp/mmfs/bin/mmlsquota -u $(whoami) --block-size G -C hk2n.scc.kit.edu hfs2-work    # workspace quota (GPFS: no `df`, no `lfs`); 250 TiB, 0.45 TB needed
squeue -u $USER                    # leftovers of earlier jobs? scancel them if you do not want them
kit_project_usage                  # granted and used hours of the project (cpu and gres/gpu)
```

* Workspaces have **no backup** and `$HOME` is 55 GiB: keep results that matter where they are backed up (`$PROJECT`, 10 TiB), at least `statistics/` of every run and the `analysis/` and `seed_*.csv` files (a few hundred MB).
* The **old HoreKA is decommissioned in December** and its file system is mounted on the HoreKA2 login nodes only (not on compute nodes): copy what you still need from it now, e.g. the in-progress
  square-box runs (the baseline of `analyse.py --baseline`, and the deck of step 1.5), with `/hfs2/data/SCC/migrate_data_hk_hk2.sh` or `rsync` on a login node.

**1.3 Code and builds**

```
cd $HST_ROOT && git status && git log --oneline -3
git branch --show-current                                   # production_filter (main has no production filter)
git merge-base --is-ancestor f4f2e69 HEAD && echo "device-memory fix present"      # git pull first if not
./compile-horeka2.sh --status      # which builds exist, from which commit, 'STALE' if src/, Makefile or env/ changed since
./compile-horeka2.sh h100 h200 cpu # (re)build; add b200 once gpu-b200 is open; no argument = all of them
```

**Every binary built before 2026-11-25 must be rebuilt**: it leaks device memory, about 27 KiB per step per rank (about 8 GiB over the 300 000 steps of a baseline-box run, 12-13 GiB over the 460 000-510 000 steps of `1e` and `3b`), fixed in `f4f2e69` (`transpose_tile_kernel`, `src/hst_mpi.f90`; results bit for bit unchanged).  Builds made by hand have no `BUILD_INFO`, so `--status` cannot tell they are old: rebuild them all.

`compile-horeka2.sh` (the repository's build routine, rewritten 2026-11-25; the file is git-ignored: copy it to HoreKA2, e.g. `rsync -a compile-horeka2.sh <login>:$HST_ROOT/`) submits **one build job per GPU type to
the matching partition** (`gpu-h100`, `gpu-h200`, `gpu-b200`), so that each binary is built on the kind of node it will run on (the B200 nodes are ARM Grace, the others x86-64), all in parallel, and waits for them; `cpu` is built where you run it.
Each job checks the node (GPU name and CPU architecture), builds into `build-gpu-<device>.new`, checks the binary's architecture, runs the repository's small deck for 20 steps on one GPU, writes
`build-gpu-<device>/BUILD_INFO` (commit, node, GPU, driver, compiler, modules) and only then replaces the old build (kept as `.old`).  A failed build leaves the previous binary alone; logs are in `build-logs/`.
Options: `--dev` (development partitions: one job at a time, 1 h limit; H200 has none), `--no-wait`, `--local <device>` (build here, as before; refused if the architecture does not match).  The job script copies `BUILD_INFO` into every run
directory (the commit of each run, as the planning document asks).

**1.4 Both campaigns side by side on the workspace** (the analysis reads `../square_seed_ensemble/seed_ensemble.csv`), and the offline checks:

```
cp -r $HST_ROOT/campaigns/square_seed_ensemble $HST_ROOT/campaigns/square_robustness $WS/
cd $WS/square_seed_ensemble   && ./check.sh decks
cd $WS/square_robustness      && python3 make_manifest.py --check manifest_document.csv && python3 make_runs.py && python3 check.py
```

(`campaigns/square_robustness` has to be in the repository on HoreKA2 first: commit and push it and `git pull` there, or `rsync -a campaigns/square_robustness/ <login>:$HST_ROOT/campaigns/square_robustness/`.)  The last two commands end with `OK`.

**1.5 Compare with your in-progress square-box deck.**  The values the document marks "to confirm" are `nx ny nz`, `cflmax`, `dt_stat`, `amplitude`, `kpeak`:

```
cd $WS/square_seed_ensemble
diff <(grep -v '^!' <your-deck>/hst.in) <(grep -v '^!' runs/square_th30_s101/hst.in)
```

Only `seed` (and `S`, `s2_amplitude` rounding) should differ.  If not, change `make_runs.py` of both campaigns, rerun it, and repeat 1.4.

**1.6 The 8-minute tests, one after the other** (`dev-gpu-h100` allows one job per user and one hour; wait for `squeue -u $USER` to be empty between them):

```
export HST_PARTITION=gpu-h100 HST_TEST_PARTITION=dev-gpu-h100

# (a) stage 0: four runs, one node
cd $WS/square_seed_ensemble && ./submit.sh test
./check.sh logs test               # four distinct seeds, first q2 about 0.11 in all four

# (b) timing pilot of the robustness grids
cd $WS/square_robustness
./submit.py --ids 1a-H9-t00 1c-R-t00 1e-B-res-t30 3b-H45-Rc6000-t00 --test
./status.py --base test ; grep -h s/step test/*/run-*.log | tail
```

Turn the s/step into the speed factor `F` (the manifest's `est_wall_h` is in the document's A100 hours; the number after the colon is the grid factor r):

```
for x in 1a-H9-t00:1 1c-R-t00:1 1e-B-res-t30:1.5 3b-H45-Rc6000-t00:1.68; do
  i=${x%%:*}; r=${x##*:}
  s=$(grep -ho '[0-9.]* s/step' test/$i/run-*.log | tail -1 | cut -d' ' -f1)
  e=$(grep "^$i," manifest.csv | cut -d, -f20)
  awk -v i=$i -v s=$s -v e=$e -v r=$r 'BEGIN{printf "%-20s %.4f s/step  speed factor %.2f\n", i, s, e/(s*304000*r/3600)}'
done
```

About 1.6 or more: the central estimates of "Cost on HoreKA2" hold; about 1.0: the conservative ones.  This is the first run of the merged code (leak fix, selective lift-up/Orr diagnostics) on a GPU of HoreKA2: the factor measured with the older binary (1.58) is superseded by this one.

The tests also log the GPU memory (every minute): `./status.py --mem` shows, per GPU, the first and last sample in MiB and the growth per hour.  The peak is the real memory footprint of that grid (replace `est_mem_gb` of the manifest by it if it differs a lot); growth above a few MiB/h would mean the leak is back.  Use the number (the smallest of the four, if they differ) as `F` below.  Repeat (b) with `HST_TEST_PARTITION=gpu-h200` if you will use it (it has no dev partition: the 8-minute job just queues).

**1.7 Two cheap checks the documentation leaves open** (a few minutes of GPU each; also run one after the other):

```
# billing: what does one GPU for two minutes cost?
kit_project_usage                                          # note the numbers
sbatch --account=$HST_ACCOUNT --partition=dev-gpu-h100 --gres=gpu:1 --time=00:05:00 --wrap="sleep 120"
sacct -j <jobid> --format=JobID,Elapsed,AllocTRES%60       # one GPU, two minutes
kit_project_usage                                          # after a few minutes (may lag): about 0.03 GPU-hours used = billed per GPU ("Billing")

# chaining: a job that submits its own continuation (sbatch from a compute node is not documented); a normal partition, not dev
cd $WS/square_robustness
HST_TEST_PARTITION=gpu-h100 MAXSEG=2 ./submit.py --ids 1a-H3-t00 --test
grep -h "segment\|resume\|MAXSEG" logs/hst-rob-test-*.out   # "segment 2 submitted", then "GPU 0: resume", then "MAXSEG = 2 segments reached"
```

*Step 1 is done when* (a) shows four distinct seeds, (b) gave `F`, and 1.7 showed the billing unit and a working chain.

### Step 2: stage 0

```
cd $WS/square_seed_ensemble
export HST_PARTITION=gpu-h100
TIME=14:00:00 ./submit.sh          # one job, four runs; 7-11 h on an H100
squeue -u $USER ; ./status.sh      # t and s/step of every run
```

A run that the limit stops is marked `INCOMPLETE` and continues in a chained job on its own.  A node that fails the pre-flight check stops the job: resubmit with `export SBATCH_EXCLUDE=<node>`.  When all four are `DONE`:

```
python3 -c "import numpy" || module spider SciPy-bundle      # the system Python has almost no packages: load a Python/SciPy-bundle module, or a venv
sbatch --account=$HST_ACCOUNT --partition=dev-cpu --time=00:20:00 --wrap="module load <the module>; cd $WS/square_seed_ensemble && python3 analyse_seeds.py > analysis.txt"
```

(The login nodes allow only short processing on a few cores, so the analysis runs as a small job, or on your machine after `rsync` of `runs/*/hst.in` and `runs/*/statistics/`.)  Read `analysis.txt`.

`seed_ensemble.csv` (and the table in `analysis.txt`): `sd` is sigma_seed, `R` the factor by which to inflate the quoted errors.

* R below about 1.5 for the main quantities (q2, S*, L_eps, b_ij): one seed per point, go on.
* R of 2 or more: `cd $WS/square_robustness && python3 make_manifest.py --replicates 2 && python3 make_runs.py` (second seeds for stages 1-2), or longer runs, `python3 make_runs.py --tmax 2000`.

*Step 2 is done when* the number of seeds per point is decided.

### Step 3: the rest of the stages

**3A: stages 1a, 1b, 1c, 1e** (13 runs).  Put in the partition with idle nodes and `F` of step 1:

```
cd $WS/square_robustness
export HST_PARTITION=<gpu-h100|gpu-h200|gpu-b200>  F=<speed factor>
./submit.py --stages 1a,1b,1c,1e --speed $F --dry-run       # read the plan: jobs, time limits, node-hours and GPU-hours
./submit.py --stages 1a,1b,1c,1e --speed $F --ack stage0
squeue -u $USER ; ./status.py --stages 1a,1b,1c,1e
```

Submitted runs are recorded in `logs/submitted.tsv` and not submitted twice (a failed run: fix it and add `--again`).  When they are `DONE`, the **review**:

```
python3 analyse.py --baseline <run dirs of the baseline box at 0, 15, 30, 45 deg>    # -> analysis/*.csv and the report
```

Read `box.csv` (1a-c, 1e against the baseline, in units of sigma_seed), `resonance.csv`, `decorrelation.csv`.  If stage 1 adopts another box than the baseline, regenerate the decks that depend on it, before anything of stage 3B runs:
`python3 make_manifest.py --adopt <A_xc> <A_yc> <A_zc> && python3 make_runs.py`.

**3B: stages 1d, 2, 3a, 3b** (17 runs).  3b is 45 % of the work.  If it goes to a B200 (from December; the nodes are ARM), build and test there first:

```
cd $HST_ROOT && ./compile-horeka2.sh --dev b200   # builds ON a B200 node (1 h limit); the first run of this code on a B200 - see build-logs/ if it fails (env/horeka2.sh: module avail NVHPC, the ARM module tree may differ)
cd $WS/square_robustness
HST_TEST_PARTITION=dev-gpu-b200 ./submit.py --ids 1a-H9-t00 3b-H45-Rc6000-t00 --test    # then the speed factor as in 1.6 -> F_B200

./submit.py --stages 1d,2,3a --speed $F --ack stage0,stage1
./submit.py --stages 3b --gpus 1 --partition <gpu-b200|gpu-h200|gpu-h100> --speed <its factor> --ack stage0,stage1
```

`--gpus 1` puts `3b-B` on one GPU (it needs about 40 GB), which is cheaper and needs no NCCL check ("Things that differ").  3a at Re_c = 1000 may go laminar:

```
./status.py --stages 3a                  # flags LAMINAR when q2 < 1 % of the baseline's and falling
./status.py --stages 3a --stop-laminar   # touch STOP: the job ends that run, the others of the job continue
```

Finally `python3 analyse.py --baseline ...` again for the orientation, edge and Reynolds tables (`analysis/*.csv`).

### Rules

* Look at `./status.py --mem` an hour into the first production job of each partition: the memory of every GPU must be flat (the fix of `f4f2e69` was verified on a small GPU with MPI transport only; NCCL and HoreKA2 are untested).
* Never submit a run directory that another job is running in (`submit.py` guards its own submissions; look at `squeue` after any manual `sbatch`).
* Keep a run and its chained continuation on one partition (the job resubmits itself into it).
* `--pack 1 --exact-gpus` (one job per run, only its GPU) is the cheap choice if the project is billed per GPU and up to four times as expensive per run if it is billed per node: decide after 1.7 ("Billing").  No `gpu-h200-8` (one node, the CUDA fault of `hkn1970`).
* Thursdays 7-9 a.m. are a regular reboot window (the docs do not say whether HoreKA2 is included): after a Thursday morning look for runs that are `started` in `./status.py` with no job in `squeue` and resubmit them with `--again`.
* Heavy work (analysis, long `rsync`/`tar`) is not for the login nodes: use `dev-cpu` jobs or your machine.  No AI agents on the login nodes.
* A run goes wrong: `tail runs/<id>/run-*.log`; `FAILED` runs are retried with `./submit.py --ids <id> --again --ack ...` (resumes from `Dati.cart.out` if there is one).

## Cost on HoreKA2 (H100, H200, B200)

**Measured (H100, `gpu-h100`, `build-gpu-h100`, 15 x 2 x 5 grid 240 x 95 x 120, four runs per node, the `horeka2_kc_scan` campaign):**
0.052 s/step in the 8-minute test and 0.049, 0.054, 0.050, 0.050 s/step in the first real job (about 0.050).  The A100 figure for the same grid is 0.079 s/step
(`doc.production/assets/README.md`, 220 000 steps = 4.5 h), so the H100 is **1.58 times faster** there.  Nothing is measured on the grids of this campaign, nor on H200 or B200.

**The planning document's constant is not consistent with that grid.**  Its 4.85 ns per point-step (4 nx ny nz points) would give 0.053 s/step on the 15 x 2 x 5 grid on an A100,
against 0.079 measured.  Two readings, hence two cases below:

* *central*: large grids run more efficiently per point, the document's constant holds for these grids on an A100, and the H100 is 1.58 times faster than the document's A100 figures;
* *conservative*: the constant is about 1.5 times optimistic; the H100 then takes about as long as the document's A100 estimate (factor 1.02: the H100's 4.6-4.8 ns per point-step, from the 0.050-0.052 s/step above, against the document's 4.85).

The in-progress square-box run (s/step on its own grid) or the 8-minute timing pilot (step 1.6) decides between them.

**Speed factors against the document's A100 figures** (`--speed`): H100 **1.58** central / 1.02 conservative (measured grid, see above); H200 **2.05** / 1.33 and B200 **3.2** / 2.0 are
*assumptions*: 1.3 and 2.0 times the H100, from the memory bandwidth (4.8 and about 8 TB/s against 3.35 TB/s for the H100; the H200 has the H100's compute), for a code that is mostly bandwidth-bound.  Measure them.

**Billing: GPU-hours or node-hours?**  The accounting page says projects are granted "CPU hours and GPU hours" and `kit_project_usage` reports `gres/gpu`; node-hours appear only for the exclusive CPU partition.
The GPU partitions are in "Shared" mode (`--gres=gpu:1` is the minimum request).  You were told node-hours.  Step 1.7 settles it with a two-minute 1-GPU job.  Consequences:

* *per GPU* (as documented): jobs should hold only the GPUs they use: `./submit.py ... --pack 1 --exact-gpus` (one job per run).  Cost = the work: 425 GPU-hours on H100 (central), 326 on H200, 212 on B200.
* *per node*: four runs per node (the default), runs of similar length together; a job holds its node until its longest run ends.  Cost = the node-hour rows below (146 on H100, 112, 73).

The estimates below assume the release order of the document (stage 0; then 1a-c with 1e; then 1d, 2, 3a, 3b), `3b-B` on one GPU (it needs about 40 GB; the cards have 94, 141, 186 GB), and one unit per
node-hour or GPU-hour on every partition (a different rate for B200 would change that row).  `./submit.py ... --dry-run` prints both costs for the runs you select.

| | H100 central | H100 conservative | H200 central | H200 conservative | B200 central | B200 conservative |
|---|---|---|---|---|---|---|
| speed factor | 1.58 | 1.02 | 2.05 | 1.33 | 3.2 | 2.0 |
| baseline run (document: 11.5 h on an A100) | 7.3 h | 11.3 h | 5.6 h | 8.7 h | 3.6 h | 5.6 h |
| longest run (3b-B, one GPU) | 58 h | 90 h | 45 h | 69 h | 29 h | 45 h |
| **node-hours, stage 0 + 1a-3b** | **146** | 226 | 112 | 174 | 73 | 113 |
| same, if no GPU idled | 106 | 165 | 82 | 127 | 53 | 82 |
| critical path, unlimited nodes (3b-B on one GPU) | 3.4 d | 5.2 d | 2.6 d | 4.0 d | 1.7 d | 2.6 d |
| critical path with 3b-B on four GPUs | 2.3 d | 3.6 d | 1.8 d | 2.7 d | 1.1 d | 1.8 d |
| node-hours with 3b-B on four GPUs | 152 | 235 | 117 | 181 | 76 | 118 |
| **GPU-hours, one job per run** (billed per GPU) | **425** | 658 | 326 | 506 | 212 | 328 |
| GPU-hours held, four runs per node-job | 584 | 904 | 449 | 696 | 292 | 452 |

(The A100 reference of the same model: 231 node-hours.  Stage 0 is 7.3 / 11.3, 5.6 / 8.7 and 3.6 / 5.6 node-hours, one node.)  The critical path is stage 0, then the
longest of 1a-c, then the longest of 1d / 2 / 3, without waiting for your review of the stages in between.  With N GPUs the duration is at least the GPU-hours of work divided by N.

Work per stage in GPU-hours (central; stage 0: 29 / 22 / 14), with the wall time of the stage's longest run:

| stage | runs | H100: GPU-h / longest | H200 | B200 |
|---|---|---|---|---|
| 1a | 6 | 43 / 16 h | 33 / 12 h | 22 / 8 h |
| 1b | 4 | 33 / 11 h | 25 / 8 h | 16 / 5 h |
| 1c | 2 | 21 / 11 h | 16 / 8 h | 11 / 5 h |
| 1d | 2 | 15 / 7 h | 11 / 6 h | 7 / 4 h |
| 1e | 1 | 37 / 37 h | 28 / 28 h | 18 / 18 h |
| 2 | 8 | 45 / 7 h | 35 / 6 h | 23 / 4 h |
| 3a | 3 | 22 / 7 h | 17 / 6 h | 11 / 4 h |
| 3b | 4 | 180 / 58 h | 139 / 45 h | 90 / 29 h |
| total | 30 | 396 | 304 | 198 |

3b is 45 % of the work (Re_c = 6000).  A node is idle about 27 % of the charged time (H100 central) because it waits for its longest run: the runs of a job are sorted by length, and moving 1e to the later group changes the total
by less than 1 %.  With `3b-B` on one GPU the four 3b runs cost 58 node-hours together on an H100 (one node, the longest run 58 h), against 71 with `3b-B` on four GPUs (two nodes of 19.6 h and one node for the H45 pair; billed per GPU with one job per run: 180 against 221 GPU-hours):
one GPU is cheaper and needs no NCCL, but takes 58 h of wall time per run instead of 32 h.  The 2-day limit holds every run in one segment in the central case except `3b-B` on one GPU (58 h, two segments);
in the conservative H100 case `1e` (57 h), `3b-H45` (50 h) and `3b-B` (90 h) need a chained second job.  Memory: every run fits one card (the largest, 3b-B, about 40 GB).

`./submit.py <stages> --dry-run --speed <factor> --partition <partition>` prints the same numbers for the runs you select: the node-hours of each job (its longest run) and of the selection, and the idle share.
To express the manifest's `est_wall_h` in H100 hours instead of A100 hours, set `NS_POINT` in `make_manifest.py` to the measured value (about 3.1e-9 in the central case, 4.7e-9 in the conservative one) and rerun it.

### After the device-memory fix and the merge of 2026-11-25

The estimates above were made with binaries from before the fix (the H100 timing is from a build of 19 Nov).  What changed in `production_filter` (`f4f2e69` leak fix, `d4ee453` merge of the selective lift-up/Orr switches):

* **Memory: the per-point figure stands, the leak is on top.**  The old binary leaked about 27 KiB of device memory per step per rank (extrapolated from 1500 steps on a small GPU): 7.8 GiB over the 304 000 steps of a baseline-box run, 11.7 GiB for `1e` (456 000 steps), 13.2 GiB for `3b` (512 000 steps) - on top of the footprint below, for every binary built before 2026-11-25.
  The footprint itself (`est_mem_gb`, 0.30 GB per million of 4 nx ny nz) is the measured peak of `examples/bench_1024.in` (nx = nz = 511, ny = 1024: 39.3 GB on each of 8 A100s = 314 GB for 1070 million points = 0.294 GB per million, FINDINGS.md "Scaling").  That run has 20 steps (the other benchmark decks 5-20),
  so the leak in it is 0.5 MB per rank, not the 39.3 GB: the number is not leak-inflated.  Production runs of the old binaries are the ones that carry the leak (the "a few GB" for the 220 000-step 15 x 2 x 5 run of `doc.production/assets/campaign/README.md` is 3.2 GB of footprint by this figure; the leak would add 5.8 GiB).
  The figure comes from 8 ranks with a distributed domain and NCCL buffers; a single GPU holding the whole domain may differ a little, which the pilot measures (`./status.py --mem`, step 1.6).  Every run fits one card (94, 141, 186 GB; the largest, `3b-B`, is 39 GB), so no decision changes.
* **Speed.**  The leak itself costs 54 small allocations per step (about 0.1-1 % of a step; not measurable without a GPU).  The merge is the larger effect: `statistics/split_runtime.dat` (the lift-up/Orr energy budget) is now computed at every statistics sample.
  CPU timing (one core, 63 x 128 x 63 grid, 40 steps; the machine is shared and some runs were 2-5 times slower under load, those are discarded; the scatter of the others is about 4 %): the code from before the merge takes 2.74 s/step at `dt_stat = 0.01` (a sample every 2-3 steps);
  the merged code 2.84-2.90 s/step at 0.01 (**4-6 % more**), 2.68 at 0.1 and 2.77 at 0.5, i.e. at `dt_stat = 0.1` it is not slower than the old code was at 0.01.  The statistics pass costs roughly 7 % of a step per sample on this grid.  On a GPU the relative cost may differ (a reduction and a host copy per sample): the pilot decides.
  So with the new `dt_stat` (below) the speed factors and the costs above stand; with the old 0.01 they would be about 5 % worse (H100 central: 446 GPU-hours instead of 425).
* **Results are unchanged.**  A campaign deck (`1a-H3-t30`, 3 steps) gives bit-identical `Runtimedata`, `variances_runtime.dat` and `production_runtime.dat` with the code from before and after the merge (default `prod_lu = prod_orr = 1`; the decks do not set them), and the leak fix changed no result.
  So the in-progress baseline runs, made with the old binary, remain comparable, and `analyse.py` (columns of those three files) needs no change.  The one new file, `statistics/split_runtime.dat`, is not used by the analysis.

### Sampling intervals: `dt_stat = 0.1`, `dt_field = 100`

From the time scales measured in the runs so far (S* = 7 and |S_tot| = 1 in all of them, so S t = t): the integral time is about one turnover, T_int = 7, and the bursting time is T_burst = 15-20 time units.

| what the analysis needs | limit | |
|---|---|---|
| autocorrelation time of q2 (about T_int = 7), block means | `dt_stat` <= T_int/10 = 0.7 | any value below |
| the bursting cycle (15-20) | `dt_stat` <= 2 | any value below |
| box-cycle lines of `<v^2>(t)`, harmonics 1-4 of the recurrence times L/(S_i L_y), the shortest 1.5 (`1a-H3`, theta = 0) | Nyquist at 4/T: `dt_stat` <= T/8 = 0.19 | **0.1** (also the grid of the spectrum in `hstlib.py`, no interpolation) |
| snapshots: independent fields are at least 2 T_burst = 40 apart | `dt_field` >= 40 | 100 is 5-7 bursts apart: independent |
| spectra windows (`exz<i>.dat`, one per `dt_field`) | enough bursts per window (a few) and several windows after t = 400 for convergence checks; the mean over all windows does not depend on `dt_field` | 100 gives 6 bursts per window and 10 windows (an upper bound of about 200 from the same argument) |

* **`dt_stat` 0.01 -> 0.1** (statistics every 22 steps instead of 2): the cost of the statistics drops from 4-6 % of the run to about 0.5 % (CPU timing above); `Runtimedata`, `variances_runtime.dat`, `production_runtime.dat` and `split_runtime.dat` shrink from 140 000 to 14 000 lines per run (35 MB to 3.5 MB each for the last).  0.1 is the smallest value
  the box-cycle test needs; if you drop the harmonics 3 and 4 (`analyse.py` tests 1-4), 0.2 would do.  Samples per independent sample: 70 per T_int, 150-200 per burst.
* **`dt_field` stays 100** (the planning document's value).  The time scales only bound it, to between 40 and about 200; inside that range the choice is about disk.  200 would halve the snapshots (7 instead of 14, the output 0.45 TB -> 0.24 TB) with no runtime gain (a snapshot is a few seconds of writing per 30 minutes to a day of run) and nothing the analysis needs,
  but it is not worth it: the workspace quota is 250 TiB, a run costs hundreds of GPU-hours and a snapshot not written cannot be recovered.  The snapshots every 100 are also intermediate restart or start fields (as the earlier fields served as initial conditions in `kc_scan`), allow the physical-space two-point correlations the planning document mentions, and give 10 spectra windows after t = 400 instead of 5 for checks of convergence.
  `dt_save` (the restart file, overwritten) stays 100: a crash loses at most 100 time units, 1/14 of a run.
* **Errors of the means** (for the blocking check): the window t >= 400 (1000 time units) holds about 70 turnovers or 30-55 bursts.  A block of 50 is 7 T_int or 2.5-3.3 bursts: the standard error should plateau between block lengths 50 and 100; if it still rises from 50 to 100, use blocks of 100 (`--block 100`).
* Both campaigns have `dt_stat = 0.1` (`template.in` of the seed ensemble and of this campaign; the in-progress square-box run still has 0.01: statistics are unaffected, only the sampling differs).  The old value: `python3 make_runs.py --dt-stat 0.01` (`check.py` then flags the difference to the seed ensemble's deck).

## The runs of a job, time limits, chaining

Single-GPU runs go four to a node (one per GPU), runs of similar length together, as a job ends when its longest run ends.  The
time limit is the longest run's `est_wall_h` / `--speed` x 1.25, at most 2 days; a longer run ends at `wall_max` with a restart file
(`Dati.cart.out`, also every `dt_save = 100`) and the job submits its continuation (`INCOMPLETE`, up to 10 jobs; which runs need it: "Cost on HoreKA2").
The plan (`./submit.py ... --dry-run --speed <factor>`) prints, per job, its node-hours (the longest run's wall hours) and GPU-hours, and for the selection the cost both ways of billing ("Cost on HoreKA2").
`--pack 1 --exact-gpus` makes one job per run asking for its GPUs only: right if the project is billed per GPU, up to four times as expensive per run if it is billed per node.
`scancel` the pending job to stop a chain.

## Things that differ from the planning document or need a decision

* **3b-B on four GPUs is not needed on H100/H200/B200.**  The document's four GPUs are for memory (about 40 GB on an A100 card).  An H100 has 94 GB, an H200 141 GB, a B200 186 GB:
  `./submit.py --ids 3b-B-Rc6000-t00 3b-B-Rc6000-t30 --gpus 1` runs each on one GPU (58 h of wall time on an H100 in the central case, 45 h on an H200, 29 h on a B200 (estimates), against
  32 / 25 / 16 h on four GPUs) and needs neither the multi-rank launch nor the NCCL check; it is also cheaper, in node-hours and in GPU-hours (see above).  `submit.py` prints this note when it applies.
  The manifest keeps the document's four GPUs: the job is then `mpirun -np 4`, ranks = GPUs, and `3b` is gated on `--ack nccl` (the document: NCCL rerun with the filter, 2.5-3x faster than MPI's alltoall).
* **First q2 and the amplitude.**  The decks use the seed decks' `amplitude = 7e-4` (as the document says).  The random field's first q2 is proportional to
  amplitude^2 Lx Lz (measured with the code: 0.027 at A_h = 3, 0.062 at 4.5, 0.113 at 6.06, 0.25 at 9, 0.167 at 9 x 6), so the starts differ by a factor 9 in energy
  between the boxes.  This matters little at Re_c = 3000 (the first 400 time units are discarded) but may at low Re_c.  `python3 make_runs.py --init-q2 0.113` sets the amplitude per
  run to give the same first q2 everywhere; the default is the document's.
* **Cost model.**  `make_manifest.py` recomputes `est_wall_h` in the document's A100 hours (4.85 ns per point-step, dt = 0.0046/r, 4 nx ny nz points, four-GPU efficiency 0.74): it reproduces the
  document's numbers except `1a-H3` (2.8 h against 3.5 h; the small grid is less efficient).  HoreKA2 has no A100: use `--speed` (above), and recalibrate the constants at the top from the in-progress run, as the
  document asks, and rerun it (and `make_runs.py` only if the grid changed).
* **Precision.**  The manifest holds the lengths, `re`, S, s2 in full precision (the document's has ten decimals): the baseline box is bit for bit that of the seed decks.
* **Deck values.**  `cflmax 0.7`, `dt_stat 0.1`, `dt_field = dt_save = 100`, `amplitude 7e-4`, `kpeak 8`, `t_max 1400` are those of `campaigns/square_seed_ensemble` (`dt_stat` differs from the in-progress run on purpose: "Sampling intervals")
  (`check.py` compares every deck with its deck in everything the manifest does not set).  Confirm them against the in-progress square-box deck (a checklist item of the document).
* **If stage 0 gives R >= 2**: more seeds per point, `python3 make_manifest.py --replicates 2` (stages 1-2 get `-r2` rows with seeds + 100), or longer runs,
  `python3 make_runs.py --tmax 2000`.  **If stage 1 adopts another box**, `python3 make_manifest.py --adopt <A_xc> <A_yc> <A_zc>` and `make_runs.py` regenerate 1d, 2-B, 3a, 3b-B
  for it (a deck whose run has output is not overwritten without `--force`).

## Stage 3a: stopping a run that has gone laminar

`./status.py --stages 3a` flags a run whose q2 is below 1 % of the baseline's (default: the mean q2 of `seed_ensemble.csv`; `--baseline-q2` sets it) after t = 450 and falling.
`./status.py --stages 3a --stop-laminar` then writes `STOP` in its directory; the job ends that run within a minute (SIGTERM, marked `STOPPED`, its last
`Dati.cart.out` kept) and the others of the job continue.  Remove `STOP` and `STOPPED` to resume it.

## Analysis (`analyse.py`)

Per run (`analysis/results.csv`, `blocks.csv`): time means for t >= 400 with block errors, the error against the block length (25, 50, 100, 200), the autocorrelation time
of q2, the means from t >= 600; q2/(S lc)^2, eps/(S^3 lc^2), S*, -<u_par v>/q2, L_eps/lc, Re_lambda, b_par, b_vv, b_perp, <u_perp v>/q2, rho_par,perp, the unforced share of q2.
Comparisons, each printed and written as a csv:

| comparison | stages | what | criterion |
|---|---|---|---|
| box convergence (`box.csv`) | 1a-c, 1e | difference to the baseline at the same angle | within 2.8 sigma_seed |
| orientation (`orientation.csv`) | all | x(30 deg) - x(0 deg) of a family; the cross terms at 30 deg against zero | within 2.8 sigma_seed |
| box-cycle resonance (`resonance.csv`) | all | lines of the `<v^2>(t)` spectrum at Lx/(S Ly), Lz/(S2 Ly) and harmonics | no line (power ratio below 3) |
| box decorrelation (`decorrelation.csv`) | all | R(dx), R(dz) from the column spectra `statistics/spectra/exz*.dat` at half the box | close to zero (flag above 0.1) |
| edge (`edge.csv`) | 2 | quantities against e at A_h = 6.06 and 4.5, and the interaction with its error | a trend |
| Reynolds number (`reynolds.csv`) | 3a, 3b | Re_lambda, S*, L_eps/lc, q2 against Re_c; whether the flow stays turbulent | a trend |

sigma_seed is the `sd` column of `../square_seed_ensemble/seed_ensemble.csv` (rescaled to the normalised quantities and at least half its quoted error);
`--inflate` multiplies the quoted errors by R where no sigma_seed applies.  The baseline is the runs given with `--baseline` plus the stage-0 runs (30 deg), found by their parameters.
The decorrelation uses the column spectra rather than the 0.7 GB snapshots: it is the correlation of u_i u_i summed over components.  The resonance test is a power ratio
against the neighbouring bins of a Welch estimate (segments of 100 time units, harmonics 1-4); a strong line also leaks into the bins next to it.
`analyse.py` and `status.py` ran on synthetic runs only (correlated noise, an injected line, a decaying run); neither has seen output of the code.

## What was tested here, what was not

Tested without a cluster: the manifest against the document (identical in mesh, seeds, lengths, re, S, s2, gpus; only `1a-H3`'s cost estimate differs),
every deck (`check.py`), one step of the CPU build for nine of the decks (all grids up to 3b-H45; 3b-B and 1e were not run), `horeka2_runs.slurm` with
stub `mpirun`/`sbatch`/`squeue`/`nvidia-smi` (single-GPU runs, a four-GPU run, two runs of two GPUs, `wall_max` chaining with the GRES string, `STOP`),
`submit.py` with a stub `sbatch` (plan, gates, double submission, test mode).  Also tested with stubs: the architecture check of `horeka2_runs.slurm` (script and matching ELF accepted, an x86 ELF on a pretended aarch64 node refused).  `compile-horeka2.sh` ran with stub `sbatch`, `nvidia-smi`, `make`, `mpirun` (all three GPU targets in parallel, `--status` with a stale build, the `.old` swap, a B200 job landing on an H100 node, a failing make that leaves the old binary, `--local b200` on x86, `--dev`, `--no-wait`); the real compile on any node has not run.  Not tested: any of it on HoreKA2, the speed of these grids (the H100 numbers above are those of the 15 x 2 x 5 grid; H200 and B200 are
estimates), the multi-GPU run (NCCL with the filter), the memory estimates (`est_mem_gb` scales the document's two numbers).  HoreKA2 runs no jobs between 2026-11-20 and about 4-7 December.

## Facts from the HoreKA2 documentation that the plan relies on (read 25 Nov 2026, docs.nhr.kit.edu)

* **Shutdown** of all clusters 20 Nov - 4 Dec 2026 (expected; the migration page's chart runs to 7 Dec): jobs only ran until noon on 20 Nov.  Workspace expiry is suspended meanwhile.  Regular reboot window every Thursday 7-9 a.m.
* **Hardware:** `gpu-h100` 22 nodes x 4 H100 (94 GB, AMD EPYC, 64 cores); `gpu-h200` 13 nodes x 4 H200 (141 GB, AMD, 96 cores); `gpu-h200-8` 1 node x 8 H200 (Intel);
  `gpu-b200` 75 nodes x 4 B200 (186 GB) on **Grace ARM CPUs (144 cores)**, "operation will start in December 2026", ARM login nodes "coming soon" (x86 login: `hk2-x86.scc.kit.edu`).
* **Slurm:** all GPU partitions "Shared", 2-day limit, `--gres=gpu:N` (minimum 1), `--partition` and `--time` mandatory, `--account` only with several projects; `MaxSubmitJobs` 100; dev partitions: 1 h, 1 node,
  one job submitted per user, "do not misuse ... for chain jobs"; examples use `mpirun`.
* **Billing:** "CPU hours and GPU hours", `kit_project_usage` (cpu, gres/gpu); no weighting per GPU type stated.
* **Storage:** GPFS; workspaces `ws_allocate name 60` (60 days, 3 extensions), 250/270 TiB and 50 million inodes in total, no backup; `$PROJECT` 10 TiB with backup; `$HOME` 50 GiB; advice: few large files, one subdirectory per process.
* **Login nodes:** only short pre/post-processing; "No compute intensive tasks"; AI agents prohibited there.  System Python 3.9/3.11/3.12 with few packages; numpy via EasyBuild modules (`Python`, `SciPy-bundle`) or a venv.
* **Not in the documentation:** the CUDA error 999 / `hkn1970` problem (ticket: https://support.nhr.kit.edu/), `sbatch` from inside a job (our chaining: step 1.7), how GPUs are isolated in shared nodes, whether H100 and H200 can share a binary.
