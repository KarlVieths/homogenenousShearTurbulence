# Next session: the production run's results

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-30, session 12)

- **The H100 is gone for good** (the user, this session): the partition
  stays reserved, HoreKA is being migrated to a new machine.  No H100
  plans; `~/hst-y/build-h100` and the stale copies `~/hst-y`, `~/hst-exp`,
  `~/hst-exp2`, `~/hst-base`, `~/hst-exp4`, `~/hst-exp5` on HoreKA can go
  (`~/hst-y` still holds the job outputs `hst-2node-517016*.out` that
  FINDINGS.md "Scaling" cites; copy them to `~/hst` first).
- **Production machinery in place** (FINDINGS.md "Production run"):
  `wall_max` in `&time_control` (the run ends after that many seconds and
  writes its restart file; decided on rank 0, broadcast), the snapshot
  and restart writes print their duration, `jobs/horeka_prod.slurm` runs
  a deck in segments and resubmits itself until `t_max`,
  `jobs/cpl_postprocess.sh` runs the CPL chain of `hst-main` on a run
  directory (Reynolds stresses and spectra per plane, budgets; the
  `hst-main` copy gets one patch, its `postprocess/` is out of step with
  `linsolver_smw.cpl`).  Checked: the CPL statistics of one of our
  fields equal our `variances_runtime.dat` line to 1e-7, the CPL pressure
  reconstruction equals our online pressure to 1e-4 outside the (0,0)
  mode.  Safety net green on CPU, GPU and NCCL after the code change.
- **Two runs were submitted at the end of the session:**
  1. `~/hst-runs/prod-small-b` (job 5171459, done): the default deck
     `hst.in` (64 x 128 x 64, Re = 1000, from q2 = 0.12) to S t = 100 on
     one A100 node in 176 s (24523 steps, 0.0072 s/step): S* = 6.09,
     -uv/q2 = 0.166, production/dissipation 1.010, Re_lambda 37, against
     WP5's 6.3 / 0.159 / 1.006 / 33 (FINDINGS.md); the CPL chain ran on it
     on the HoreKA login node.  The first attempt (job 5171453)
     relaminarized from the old weak initial field.  `prod-small-c` (job
     5171488 and successors, `MARGIN=800` so that the segments are 100 s)
     was the same run as a chain of two segments: the restart chain
     works end to end (FINDINGS.md), S* = 6.17, -uv/q2 = 0.165.
  2. `/hkfs/work/workspace/scratch/xt8786-hst/re20000`: the production
     deck `examples/prod_re20000.in` (1536 x 1024 x 512, Re = 20000, dx =
     dy = dz, dx/eta = 1.2 estimated, Re_lambda about 150) to S t = 100 on
     two A100 nodes, job 5171460 on `accelerated`, `--time=12:00:00`
     (expected 8-12 h of compute at 0.33-0.45 s/step; the two-node queue
     took 6-16 h in the last sessions).  The chain continues by itself
     if 12 h are not enough.  Snapshots and restarts every 5 time units
     (11.5 GB per snapshot, 230 GB in all; the workspace expires
     2026-11-29, `ws_extend hst 30` if needed).

## The task: read the production run

1. **The small runs are done** (`prod-small-b`, `prod-small-c`): chain,
   statistics and CPL post-processing all checked (FINDINGS.md).  The
   post-processing recipe on HoreKA: `module load compiler/gnu/13`, then
   separately `module load mpi/openmpi/5.0`, `~/.local/bin` on the PATH
   for `cpl`, `HST_MAIN=~/hst-main jobs/cpl_postprocess.sh <run dir>
   <ranks> [nfmin nfmax dn]`.  Left open: the restart repeats one
   `Runtimedata` line (cosmetic), and the statistics reduction on the
   device is not bit-reproducible between runs (the fields are).
   Its columns are `time meanflowx meanflowy S S2 gamma_x gamma_y deltat
   cfl energy diss uw/2 vw/2`, integrals over ly = 2, so `energy` = <q2>,
   `diss` = <grad u : grad u> (without nu), `uw/2` = <u v>: eps =
   diss/re, S* = S energy re/diss, -uv/q2 = -(uw/2)/energy, P/eps = -S
   (uw/2) re/diss; compare with FINDINGS.md "Long sheared run".
2. **Read the production run's first segment**: s/step (the deck's cost
   was estimated, not measured: a third of the 1024^3 points), the
   snapshot and restart-file times at 8.6 + 2.9 GB, the memory (no `mem`
   marker in this script; `nvidia-smi` is not on the login node), the
   CFL step and the number of steps to S t = 100 it implies, whether one
   segment was enough.  Put the numbers into FINDINGS.md "Production
   run" and the README performance paragraph.
3. **Post-process it** when it reaches S t = 100 (or earlier on the
   snapshots so far): `jobs/cpl_postprocess.sh <run dir> <ranks>` on a
   CPU node (`postpro.cpl` holds one whole field per rank: 8.6 GB, plus
   the spectra; `cpuonly` nodes have 256 GB), `nfmin` after the
   transient.  Then the statistics: S*, -uv/q2, the anisotropy, Re_lambda,
   the resolution actually achieved (eta from diss: if dx/eta came out
   far from 1.2 the deck's Re is the knob), the spectra.  Whether the
   time step, the snapshot cadence and the segment length were right is
   the point of the first run.
4. **If the run misbehaves** (out of memory on 8 ranks is impossible at
   13 GB, but the two-slab layout at nz = 170 is new): `scancel` the
   chain, run the deck's 8-rank layout for 20 steps with
   `jobs/horeka_2node.slurm` (`CONFIGS="8:prod_re20000:20:nccl:none:0:mem"`)
   and look.

Later items, unchanged from session 11: the reduced-system gather for
`npy > 2` (FINDINGS.md "Scaling": worth 18% of the four-node 1024^3
step, nothing at two nodes; lever (d) or an alltoall of the records
within the y column), the cleanup of the stale HoreKA copies, and the
I/O (185 MB/s for the 25.8 GB restart file at 1024^3, 1 s of latency per
snapshot at any size: ROMIO hints or the workspace file system, if the
production run's snapshot times say it matters).

## Safety net (the third argument is npy)

```bash
tests/run_tests.sh build-cpu 2 && tests/run_tests.sh build-cpu 4 2 && tests/run_tests.sh build-cpu 4 4   # 12 runs each
tests/run_tests.sh build-gpu 2 && tests/run_tests.sh build-gpu 2 2
tests/regression.sh build-cpu 2 && tests/regression.sh build-cpu 4 2        # 50 steps of three decks at 1e-10
tests/regression.sh build-gpu 1 && tests/regression.sh build-gpu 2 && tests/regression.sh build-gpu 2 2
tests/crossbranch.sh build-cpu build-cpu 4 1 2 && tests/crossbranch.sh build-cpu build-cpu 4 2 1   # restart between npy values
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && make GPU=1 NCCL=1 BUILD=build-nccl -j8 && make GPU=1 NCCL=1 BUILD=build-nccl test -j8 && tests/run_tests.sh build-nccl 2 2 && tests/regression.sh build-nccl 2 2 && tests/run_tests.sh build-nccl 2 && tests/regression.sh build-nccl 2 && tests/crossbranch.sh build-nccl build-nccl 2 1 2'
```

The CPU build must be made in a shell that has *not* sourced
`env/istm.sh`, the GPU tests in one that *has*; `make test` rebuilds the
test programs; a flag change needs the objects removed.  Parallel Bash
calls share one working directory: absolute paths or `make -C`.  A
bit-identity check of a change that must not alter the numbers: the
`small` deck with the GPU build before and after (50 steps), `cmp` the
two `Dati.cart.out`.  nvfortran 25.9 rejects the names `kind` and `x` in
OpenMP clauses of any file that can see `hst_mpi`; an absent optional
dummy must not appear in a target region.  HoreKA: `rsync -a --delete
--exclude 'build-*' --exclude .git --exclude 'hst-*.out' ./ horeka:hst/`,
then `source env/horeka.sh gpu; make GPU=1 GPU_ARCH=cc80 NCCL=1`; never
rebuild a root while a job is queued on it (the production chain reads
`~/hst/build-gpu/hst` at every segment: rebuild `~/hst` only between
segments, or point the chain at a copy with `HST_ROOT`).  Scripted ssh to
HoreKA: `ssh -o BatchMode=yes -o ProxyCommand=false horeka ...`, one call
at a time; `ssh -O check horeka` says whether the master is alive; when
it is not, the user types `! ssh horeka true`.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (one branch, main; on
HoreKA ~/hst = main), a GPU/CPU DNS for homogeneous shear turbulence with
x-z pencils times npy y slabs; read README.md, then NEXT_SESSION.md, then
FINDINGS.md from "Production run" to the end.  Task: NEXT_SESSION.md items
2-3 (the production run's numbers, its post-processing with the CPL
chain) unless I say otherwise.  Safety net
green after any code change (CPU, GPU, NCCL on istmcetus, the npy restart
round trip).  Do not modify ~/Codes/hst/channel or ~/Codes/hst/hst-main.
Commit each step; push at the end.  First thing: `! ssh horeka true` in
the prompt if `ssh -O check horeka` says the master is gone.
```
