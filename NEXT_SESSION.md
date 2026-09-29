# Next session: beyond the y exchange

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-29, session 10)

Both branches, pushed, safety net green, HoreKA copies rebuilt (`~/hst`
= `main`, `~/hst-y` = `~/hst-exp` = `~/hst-exp2` = the branch;
`~/hst-base`, `~/hst-exp4`, `~/hst-exp5` are older variant copies from
earlier A/B jobs and can go, `~/hst-exp3` went this session):

- **The two branches stay separate and run the same decks and field
  files.**  `tests/crossbranch.sh <build A> <build B> [nranks] [npy A]
  [npy B]` (on `main`) proves it: `main` <-> the branch at `npy = 2`,
  both directions, 2e-14 CPU / 6e-14 NCCL (checked again this session).
- **`npy = 0` (the default) means "the code chooses"**: one slab on
  `main`; on the branch one slab per node when the nodes are equal and
  node-consecutive, at most 8, and the layout fits the grid.
- **Lever (b), one ghost exchange for u and w, is in (FINDINGS.md, last
  section).**  `fill_ghosts(c, c2)` and `exchange_ghost_rows(field,
  shift_x, shift_z, field2)` take an optional second component / field;
  `linsolve` ends with `call fill_ghosts(1, 3)`, 9 instead of 12 ghost
  exchanges per step.  On `main` the wrap runs once per field (internal
  subroutine `wrap_rows`); on the branch the buffers are `(2, z, x,
  field, direction)`, `pack_rows`/`unpack_rows` run once per present
  field and one message per direction carries both.  Bit-identical
  (`cmp` of `Dati.cart.out`, `main` GPU 1 and 2 ranks, branch 1 x 2 via
  MPI and NCCL).  Job 5170040 (A/B, same two nodes): 8 GPU 256^3 0.02307
  -> 0.02262 (2.0%), 512^3 0.12178 -> 0.12118 (0.5%), `small` 0.00784 ->
  0.00703 with its ghost rows down 27% (the predicted quarter); 4-GPU
  rows a wash.  DESIGN.md's departures list has the line (the first
  change to a physics file made for the parallel layer; contract 7 (i)
  item 2 names the new signature).
- **The y exchange after (a) dropped, (b) and (c) kept:** 8.65 ms ghost
  rows + 4.11 ms exposed records of 121.2 ms at 512^3 (10.5%), 4.93 +
  3.42 of 22.6 ms at 256^3.  Two nodes = 1.33x one node at 256^3, 1.79x
  at 512^3.  The cheap levers are used up: (d) smaller records is bytes
  only (<= 3% at 512^3); the ghost rows' fixed cost (0.2-0.26 ms per
  inter-node exchange, 9 + 9 exchanges per step) can only be hidden by
  computing interior rows while an exchange is in flight, a change to
  the physics files' loop structure (each y-stencil kernel split into
  interior and boundary rows, the exchange between them), i.e. not a
  parallel-layer change any more.

`jobs/horeka_2node.slurm`: `CONFIGS` entries `ranks:deck:steps:
transport:bind:npy[:info|:lc<N>]`, a deck of `examples/` or
`tests/decks/`, `ROOTS="a b"` for an in-job A/B, `RUNS` for the run
directory (two jobs of the same configs must not share one), the banner
of every run printed; time limits are the expected length plus a margin
(the 2-root A/B of 5 configs took 5.1 min in 12).  Submitting the same
job on `accelerated` and `dev_accelerated` and cancelling the loser
worked this session: the `dev_accelerated` one started after 50 min
(the `accelerated` one was still pending).

## The task: choose the next chapter (the user decides; a recommendation)

The exchange chapter is closed on the cheap side.  The candidates, in
the order the handoff of session 9 listed them, with what each costs:

1. **Scaling beyond two nodes (measurement only, no code).**  A four-node
   run at 512^3 (`--nodes=4 --ntasks-per-node=4`, `npy = 0` gives 4 x 4)
   and, since the interesting production sizes are larger, 1024^3 on
   two and four nodes (`examples/bench_1024.in` does not exist yet: copy
   `bench_512.in` and double the modes; memory first: 512^3 on 8 ranks
   uses 293 MB of cuFFT work area and 2 x 199 MB of solver workspace per
   rank plus the fields, 1024^3 has 8x the points, so run it on one
   node's 4 ranks with `nstep = 1` and read the banner and `nvidia-smi`
   before the multi-node jobs).  Expected: four nodes at 512^3
   well below 2x two nodes (the 13 ms of exchange stay, the compute
   halves); at 1024^3 the exchange fraction is lower and the picture
   better.  Also the H100 partition (`accelerated-h100`, `GPU_ARCH=cc90`,
   its own build dir `BUILD=build-h100`), where the compute per node is
   about 2-3x and the InfiniBand the same, so the two-node ratio drops.
   This tells whether the code is production-ready at the sizes the
   user wants, and is the recommended next step.
2. **Lever (d), smaller records (<= 3% at 512^3).**  For the system kinds
   whose matrix does not change between calls (`KIND_DY`, `KIND_D0INV`,
   the Poisson kind at fixed k2 changes with the mode only), the reduced
   system's coefficients are the same every step and only the right-hand
   side's records need gathering: half the bytes of those gathers.  Bytes
   only; measured 4.11 ms exposed records at 512^3 in total, of which the
   solves with fixed matrices are a part.  Not worth a session alone.
3. **Merging the branch into `main`.**  The branch differs in `hst_mpi`
   and `hst_linsolve` only (629 insertions).  With `npy = 0` the merged
   code behaves as `main` on one node and as the branch on several; the
   price is that `main` carries the reduced-system solver and the second
   communicator.  The user has kept the branches separate so far so that
   `main` stays the simplest reading of the code; merging is a decision,
   not a task, and would be done as `git merge multinode-y` on `main`
   with the safety net, then `tests/crossbranch.sh` becomes a restart
   test between `npy` values.
4. **Production runs** with the CPL post-processing on the output
   (README, "Output"): a run to S t = 100 at a size of the user's
   choosing, `Runtimedata` and the fields checked with
   `hst-main/postprocess`.

If the user has not said otherwise, do 1: the scaling measurements at
512^3 and 1024^3 on two and four A100 nodes and on two H100 nodes, one
job per partition, `--time` the expected length plus a margin, results
in a FINDINGS section and the README's performance table (a 4 x 4 A100
column and an H100 column), no code change unless a run fails.

## Safety net (both branches; the branch adds a third argument, npy)

```bash
tests/run_tests.sh build-cpu 2         # 12 runs; on the branch also: tests/run_tests.sh build-cpu 4 2  and  ... 4 4
tests/run_tests.sh build-gpu 2         #                              tests/run_tests.sh build-gpu 2 2
tests/regression.sh build-cpu 2        # 50 steps of three decks at 1e-10; branch: tests/regression.sh build-cpu 4 2
tests/regression.sh build-gpu 1
tests/regression.sh build-gpu 2        #                                       branch: tests/regression.sh build-gpu 2 2
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && make GPU=1 NCCL=1 BUILD=build-nccl -j8 && make GPU=1 NCCL=1 BUILD=build-nccl test -j8 && tests/run_tests.sh build-nccl 2 2 && tests/regression.sh build-nccl 2 2 && tests/run_tests.sh build-nccl 2 && tests/regression.sh build-nccl 2'
cd ../homogenenousShearTurbulence-main && tests/crossbranch.sh build-cpu ../homogenenousShearTurbulence/build-cpu 4 1 2   # and the reverse; on istmcetus with build-nccl and 2 ranks
```

A bit-identity check of a change that must not alter the numbers: run
the `small` deck with the GPU build before and after (50 steps, the
regression's run) and `cmp` the two `Dati.cart.out` (the GPU runs are
deterministic, the CPU build with FFTW_MEASURE is not; `main` at 1 and
2 ranks and the branch at 1 x 2 via MPI or NCCL each write the same
bytes, `main` and the branch differ at 1e-14).

Worktrees: `~/Codes/hst/homogenenousShearTurbulence` = `multinode-y`,
`~/Codes/hst/homogenenousShearTurbulence-main` = `main` (`git worktree
list`); a variant to measure goes into its own worktree and branch,
rsynced to a `~/hst-exp<n>` root on HoreKA (remove its stale `build-gpu`
first) and run with `ROOTS` in one two-node job.  The CPU build must be
made in a shell that has *not* sourced `env/istm.sh`, the GPU tests in
one that *has*; a flag change needs the objects removed; `build-nccl` is
exercised on istmcetus only, where `2 2` (one pencil x two slabs) uses
the NCCL y column and `2` the NCCL alltoall.  Decks with several solver
batches: `line_chunk = 3` or `5` in `&mesh`.  nvfortran 25.9 rejects the
names `kind` and `x` in OpenMP clauses of any file that can see
`hst_mpi`; an absent optional dummy must not appear in a target region
(test `present` on the host, launch per field).  Not on `main`: the
`buildrhs` tile and anything that adds a kernel for a few percent.
Scripted ssh to HoreKA: `ssh -o BatchMode=yes -o ProxyCommand=false
horeka ...`, one call at a time (parallel mux sessions are refused);
`ssh -O check horeka` says whether the master is alive; when it is not,
the user types `! ssh horeka true`.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (branch multinode-y;
~/Codes/hst/homogenenousShearTurbulence-main is a worktree of main; on
HoreKA ~/hst = main, ~/hst-y = ~/hst-exp = ~/hst-exp2 = the branch), a
GPU/CPU DNS for homogeneous shear turbulence; read README.md, then
NEXT_SESSION.md, then FINDINGS.md from "The fixed cost of the y
exchange" to the end.  The two branches stay separate and run the same
decks and field files; branches differ only in hst_mpi and
hst_linsolve; merge main into the branch, never the reverse; docs on
main.  Task: item 1 of NEXT_SESSION.md unless I say otherwise: the
scaling measurements of the branch at 512^3 and 1024^3 on two and four
A100 nodes and on two H100 nodes (jobs/horeka_2node.slurm, one job per
partition, --time the expected length plus a margin), no code change
unless a run fails; then FINDINGS, the README's performance table and
NEXT_SESSION.  Safety net green after any code change (CPU, GPU, NCCL
on istmcetus, the cross-branch round trip).  Do not modify
~/Codes/hst/channel or ~/Codes/hst/hst-main.  Commit each step; push
both branches at the end.  First thing: `! ssh horeka true` in the
prompt if `ssh -O check horeka` says the master is gone.
```
