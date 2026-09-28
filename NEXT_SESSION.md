# Next session: what is left of the y exchange, or something else

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-29, session 9)

Both branches, pushed, safety net green, HoreKA copies rebuilt (`~/hst`
= `main`, `~/hst-y` = `~/hst-exp` = the branch; `~/hst-exp2` and
`~/hst-exp3` are stale variant copies from the A/B jobs and can go):

- **The two branches stay separate and run the same decks and field
  files.**  `tests/crossbranch.sh <build A> <build B> [nranks] [npy A]
  [npy B]` (on `main`) proves it: `main` <-> the branch at `npy = 2`,
  both directions, 2e-14 CPU / 6e-14 GPU.
- **`npy = 0` (the default) means "the code chooses"**: one slab on
  `main`; on the branch one slab per node (ranks per node from
  `MPI_Comm_split_type` in `setup_decomposition`) when the nodes are
  equal and node-consecutive in `MPI_COMM_WORLD`, at most 8, and the
  layout fits the grid, else one slab with a notice; an explicit `npy >
  1` on non-consecutive ranks aborts.  Verified on istmcetus, across
  istmio2 + istmcetus with the system OpenMPI, and on two HoreKA nodes
  ("ranks = 8 (4 x-z pencils x 2 y slabs)", regressions at 5.9e-14).
  No launcher scripts; the Slurm jobs are unchanged.
- **The y exchange, measured (FINDINGS.md, the last four sections):**
  the fixed cost of an inter-node exchange is 0.2-0.26 ms before any
  bytes (the `small` deck on 8 ranks), five times NVLink's.  Lever (a),
  one NCCL group for the ghost rows: a wash to worse, dropped.  Lever
  (c), the records' gather on the communication stream behind the next
  solver batch's sweep (two workspaces, two batches by default when each
  fills the GPU, `LINES_FULL` in `hst_linsolve`; `line_chunk` still
  overrides): kept, 512^3 on 2 x 4 A100 0.1270 -> 0.1218 s/step, one
  node a wash.  Two nodes are now 1.29x one node at 256^3 and 1.77x at
  512^3.  What is left in the y exchange: 8.6 ms ghost rows + 4.6 ms
  exposed records of the 121.8 ms step at 512^3 (11%), 4.7 + 1.6 of
  23.6 ms at 256^3 (27%, mostly the fixed cost).

`jobs/horeka_2node.slurm`: `CONFIGS` entries `ranks:deck:steps:
transport:bind:npy[:info|:lc<N>]`, a deck of `examples/` or
`tests/decks/`, `ROOTS="a b"` for an in-job A/B, the banner of every run
printed; time limits are the expected length plus a margin (a 2-root
A/B of 7 configs took 7.6 min): shorter jobs backfill sooner.

## Levers left, with what they can buy

- (b) The two ghost exchanges that follow each other in `hst_equations`
  (`fill_ghosts(1)` then `fill_ghosts(3)` after the recovery of u and w)
  as one exchange of two fields: 12 -> 9 exchanges a step, i.e. 3 x
  (0.2-0.4 ms) = about 1 ms of 23.6 at 256^3 (4%), 2 of 122 at 512^3.
  Needs a `fill_ghosts(c1, c2)` in `hst_derivatives` (pack two
  components into one buffer) and the two calls in `hst_equations`:
  physics files on `main`, so **the user decides first**.
- (d) Smaller records for the system kinds whose matrix does not change
  between calls (`KIND_DY` every substep, the implicit kinds while
  `deltat` is fixed): bytes only; with (c) the exposed record time at
  512^3 is 4.6 ms of 122, so at most 3%.
- The ghost rows' fixed cost (12 x 0.2-0.4 ms) is out of reach without
  computing the interior rows while the exchange is in flight, which is
  structure in the physics files that `main` does not want.
- Not levers of the exchange but of the step: one node at 512^3 is
  0.216 s on 4 A100 (FINDINGS "The line solver's latency" and "The
  kernels around the transposes" have the per-kernel picture); the
  H100 partitions (`accelerated-h100`, `cc90`) have not been tried.

So the y decomposition is at the point where the remaining exchange
levers are worth a few percent each; whether to spend a session on (b)
+ (d), on the H100 nodes, on production runs (long sheared runs,
statistics, the CPL post-processing chain on the files), or on merging
the branch into `main` (the user's call; today `main` = x-z pencils
only, and the branch differs in `hst_mpi`, `hst_linsolve` and one job
script) is the question for the user at the start of the next session.

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

Worktrees: `~/Codes/hst/homogenenousShearTurbulence` = `multinode-y`,
`~/Codes/hst/homogenenousShearTurbulence-main` = `main` (`git worktree
list`); a variant to measure goes into its own worktree and branch,
rsynced to a `~/hst-exp<n>` root on HoreKA and run with `ROOTS` in one
two-node job.  The CPU build must be made in a shell that has *not*
sourced `env/istm.sh`, the GPU tests in one that *has*; a flag change
needs the objects removed; `build-nccl` is exercised on istmcetus only,
where `2 2` (one pencil x two slabs) uses the NCCL y column and `2` the
NCCL alltoall.  Decks with several solver batches: `line_chunk = 3` or
`5` in `&mesh`.  A bit-for-bit CPU comparison needs `FFTW_ESTIMATE`.
nvfortran 25.9 rejects the names `kind` and `x` in OpenMP clauses of any
file that can see `hst_mpi`.  Not on `main`: the `buildrhs` tile and
anything that adds a kernel for a few percent.  Scripted ssh to HoreKA:
`ssh -o BatchMode=yes -o ProxyCommand=false horeka ...`, one call at a
time (parallel mux sessions are refused); `ssh -O check horeka` says
whether the master is alive; when it is not, the user types `! ssh
horeka true`.  Two-node jobs on `accelerated` waited 11-15 h on
2026-09-28.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (branch multinode-y;
~/Codes/hst/homogenenousShearTurbulence-main is a worktree of main; on
HoreKA ~/hst = main, ~/hst-y = ~/hst-exp = the branch), a GPU/CPU DNS
for homogeneous shear turbulence; read README.md, then NEXT_SESSION.md,
then FINDINGS.md from "The layout without user input" to the end.  The
two branches stay separate and run the same decks and field files;
branches differ only in hst_mpi and hst_linsolve; merge main into the
branch, never the reverse; docs on main.  The y decomposition's exchange
levers are measured; NEXT_SESSION.md lists what is left and what each
can buy.  Task: [the user fills in: (b) + (d), the H100 nodes,
production runs, or merging the branch].  Safety net green after every
code change (CPU, GPU, NCCL on istmcetus, the cross-branch round trip);
HoreKA A/Bs in one two-node job with ROOTS, time limit = expected length
plus margin.  Do not modify ~/Codes/hst/channel or ~/Codes/hst/hst-main.
Commit each step; push both branches at the end.  First thing: `! ssh
horeka true` in the prompt.
```
