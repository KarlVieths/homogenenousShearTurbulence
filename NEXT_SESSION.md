# Next session: production, or the H100

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-29, session 11)

- **One branch.**  `multinode-y` was merged into `main` this session
  (`git merge --no-ff`, FINDINGS.md "The merge"): tag `xz-parallel` is
  `main` before the merge (x-z pencils only), tag `xyz-parallel` the
  merge commit.  `main` runs `npy` y slabs, `npy = 0` (the default)
  choosing one slab per node when that fits the grid; the safety net was
  green on CPU, GPU and NCCL after the merge and `tests/crossbranch.sh
  <build> <build> 4 1 2` (and `4 2 1`) is now the restart test between
  `npy` values.  The branch `multinode-y` is deleted (locally and on
  origin) and the `-main` worktree removed: one checkout,
  `~/Codes/hst/homogenenousShearTurbulence`, on `main`.  On HoreKA `~/hst` is
  the merged `main` (rebuilt); `~/hst-y`, `~/hst-exp`, `~/hst-exp2` are
  the branch before the docs (same code) and `~/hst-base`, `~/hst-exp4`,
  `~/hst-exp5` older variants: all five can go.
- **Scaling (FINDINGS.md "Scaling", README performance table).**  Two
  A100 nodes: 256^3 0.02257 s/step (1.34x one node), 512^3 0.12089
  (1.79x), 1024^3 1.005 s/step at 39.3 GB of the 40 GB per GPU (four
  ranks run out of memory: 1024^3 needs two nodes and has no one-node
  reference on the A100).  At 1024^3 the y exchange is 2.5% of the step
  (10.8% at 512^3), so the remaining exchange levers (interior/boundary
  row splitting, lever (d)) are worth at most that at production size.
  Four nodes: job 5170168 (`accelerated`, `--nodes=4`, 512^3 and
  1024^3 on 16 GPUs with the same-job 8-GPU rows, `RUNS=~/hst-runs/
  scal-a4`) was still pending at the end of the session; when it has
  run, `~/hst-y/hst-2node-5170168.out` holds the rows: add them to
  FINDINGS.md "Scaling" and a "4 x 4 A100" column to the README table.
  Expected: well below 2x two nodes at 512^3 (the exchange stays, the
  compute halves), closer to 2x at 1024^3.
- **H100 not measured**: the whole `accelerated-h100` partition is in
  the reservation `hk2teal` until 2026-10-31.  `~/hst-y/build-h100` is
  built (`GPU_ARCH=cc90`) and job 5170169 (two H100 nodes, the two-node
  configs, `RUNS=~/hst-runs/scal-h2`) is left pending; it runs by itself
  if the nodes come back, otherwise `scancel` it.  When the partition
  opens: rebuild from `~/hst` (`make GPU=1 GPU_ARCH=cc90 NCCL=1
  BUILD=build-h100`) and submit `jobs/horeka_2node.slurm` with
  `BUILD=build-h100` (header of the script); whether 1024^3 fits on one
  H100 node (estimated 75 GB per GPU) is part of the question.

`jobs/horeka_2node.slurm`: `CONFIGS` entries `ranks:deck:steps:
transport:bind:npy[:info|:lc<N>|:mem]` (`mem` prints the peak GPU memory
of the run), `BUILD` for another build directory, `--nodes=4` on the
sbatch line for four nodes, `ROOTS="a b"` for an in-job A/B, `RUNS` per
job; `--time` the expected length plus a margin (two nodes, six configs
with 1024^3: 6 min in 12).  Submitting the same two-node job on
`accelerated` and `dev_accelerated` and cancelling the loser worked
again (the `dev_accelerated` one started after 28 min).

## The task: choose (the user decides; a recommendation)

1. **A production run** with the CPL post-processing on the output
   (README, "Output"): a run to S t = 100 at a size of the user's
   choosing, `Runtimedata` and the fields checked with
   `hst-main/postprocess`.  1024^3 costs 1.0 s/step on two A100 nodes;
   the time step and the number of steps to S t = 100 follow from the
   CFL of the deck.  Recommended: it is what the code is for, and the
   first run will find what the benchmarks do not (restart cadence,
   field I/O time at 1024^3, the statistics).
2. **Cleanup after the merge**: remove the stale HoreKA copies, fold `tests/crossbranch.sh` into the safety net
   as the `npy` restart test (README "Tests"), retire the "branch"
   wording that is left in FINDINGS.md's older sections (they are
   history, so probably leave them).
3. **Hiding the ghost rows' fixed cost** (interior/boundary row
   splitting in the y-stencil kernels): at most 10% at 512^3 on two
   nodes, 2.5% at 1024^3.  Not recommended.

## Safety net (one branch now; the third argument is npy)

```bash
tests/run_tests.sh build-cpu 2 && tests/run_tests.sh build-cpu 4 2 && tests/run_tests.sh build-cpu 4 4   # 12 runs each
tests/run_tests.sh build-gpu 2 && tests/run_tests.sh build-gpu 2 2
tests/regression.sh build-cpu 2 && tests/regression.sh build-cpu 4 2        # 50 steps of three decks at 1e-10
tests/regression.sh build-gpu 1 && tests/regression.sh build-gpu 2 && tests/regression.sh build-gpu 2 2
tests/crossbranch.sh build-cpu build-cpu 4 1 2 && tests/crossbranch.sh build-cpu build-cpu 4 2 1   # restart between npy values
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && make GPU=1 NCCL=1 BUILD=build-nccl -j8 && make GPU=1 NCCL=1 BUILD=build-nccl test -j8 && tests/run_tests.sh build-nccl 2 2 && tests/regression.sh build-nccl 2 2 && tests/run_tests.sh build-nccl 2 && tests/regression.sh build-nccl 2 && tests/crossbranch.sh build-nccl build-nccl 2 1 2'
```

A bit-identity check of a change that must not alter the numbers: run
the `small` deck with the GPU build before and after (50 steps) and
`cmp` the two `Dati.cart.out` (the GPU runs are deterministic, the CPU
build with FFTW_MEASURE is not).  The CPU build must be made in a shell
that has *not* sourced `env/istm.sh`, the GPU tests in one that *has*;
a flag change needs the objects removed.  Decks with several solver
batches: `line_chunk = 3` or `5` in `&mesh`.  nvfortran 25.9 rejects the
names `kind` and `x` in OpenMP clauses of any file that can see
`hst_mpi`; an absent optional dummy must not appear in a target region
(test `present` on the host, launch per field).  Not on `main`: the
`buildrhs` tile and anything that adds a kernel for a few percent.
HoreKA: `rsync -a --delete --exclude 'build-*' --exclude .git --exclude
'hst-*.out' ./ horeka:hst/`, then `source env/horeka.sh gpu; make GPU=1
GPU_ARCH=cc80 NCCL=1`; never rebuild a root while a job is queued on it.
Scripted ssh to HoreKA: `ssh -o BatchMode=yes -o ProxyCommand=false
horeka ...`, one call at a time (parallel mux sessions are refused);
`ssh -O check horeka` says whether the master is alive; when it is not,
the user types `! ssh horeka true`.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (one branch, main,
tags xz-parallel / xyz-parallel; on HoreKA ~/hst = main), a GPU/CPU DNS for homogeneous shear turbulence with x-z pencils
times npy y slabs; read README.md, then NEXT_SESSION.md, then
FINDINGS.md from "Scaling" to the end.  Task: item 1 of NEXT_SESSION.md
unless I say otherwise.  Safety net green after any code change (CPU,
GPU, NCCL on istmcetus, the npy restart round trip).  Do not modify
~/Codes/hst/channel or ~/Codes/hst/hst-main.  Commit each step; push at
the end.  First thing: `! ssh horeka true` in the prompt if `ssh -O
check horeka` says the master is gone.
```
