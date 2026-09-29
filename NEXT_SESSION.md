# Next session: lever (b), one ghost exchange for two fields

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

## The task: lever (b), one exchange for two fields (the user's decision, 2026-09-29)

The recovery of u and w at the end of `linsolve` (`hst_equations`, the
last lines) ends with `call fill_ghosts(1)` and `call fill_ghosts(3)`:
two ghost-row exchanges back to back, each paying the inter-node fixed
cost (0.2-0.4 ms) on top of its bytes.  As one exchange of two fields
the step has 9 instead of 12 exchanges: about 1 ms of 23.6 at 256^3
(4%) and 2 ms of 122 at 512^3 (under 2%), nothing on one node.  The
other exchanges of a substep stay separate: `fill_ghosts(2)` feeds the
`KIND_DY` solve that follows it, `fill_ghosts_field(V(:, :, :, 1), sx1,
sz1)` in `shear_shift` uses other displacements, and `fill_ghosts(3)`
after `stokes_apply` runs only with a Stokes layer.  This is the first
change to a physics file made for the parallel layer, so it goes on
`main` in the smallest form and reaches the branch by `git merge main`
(the branch then changes `hst_mpi` only), and DESIGN.md's list of
departures gets a line.

The change, about ten lines longer in all:

- `hst_derivatives`: `fill_ghosts(c, c2)` with an optional second
  component, passing `V(:, :, :, c2)` on to `exchange_ghost_rows` as an
  optional fourth argument; `fill_ghosts_field` (one field, used by
  `shear_shift` and `test_linsolve`) unchanged.
- `hst_equations`: the two calls become `call fill_ghosts(1, 3)`.
- `hst_mpi` on `main`: `exchange_ghost_rows(field, shift_x, shift_z,
  field2)`, the present in-place wrap as an internal subroutine called
  for `field` and, if present, `field2` (an absent optional must not
  appear inside a target region: test `present` on the host and launch
  twice).
- `hst_mpi` on the branch: the same signature; the buffers
  `ghost_send`/`ghost_recv` get a field dimension of 2, the pack and the
  unpack become internal subroutines `pack(field, f)` / `unpack(field,
  f)` launched once per present field, and the NCCL/MPI transfer sends
  `nf` times the bytes as one message per direction (the buffer layout
  `(2, -nz:nz, nx0:nxN, f, direction)` keeps a direction's blocks
  contiguous when `f` is the inner of the two).  The `npy = 1` wrap
  handles both fields as on `main`.
- Results must be bit-identical to today's (the same operations in the
  same order): besides the safety net, compare `Dati.cart.out` of the
  `small` deck on istmcetus (`build-nccl`, 2 ranks, `npy = 2`) before
  and after with `cmp`, and the two-node regression decks at `npy = 0`
  in the job below.
- The A/B: worktree + branch `lever-b` as before, rsync to `~/hst-exp2`
  (a stale copy today; `~/hst-exp3` too), build, then from `~/hst-exp`

  ```bash
  ssh horeka 'cd ~/hst-exp && sbatch --parsable --partition=accelerated --time=00:12:00 --export=ALL,HST_ROOT=$HOME/hst-exp,ROOTS="$HOME/hst-exp $HOME/hst-exp2",NPY_REG=0,CONFIGS="4:bench_256:20:nccl:none:0 8:bench_256:20:nccl:none:0 4:bench_512:10:nccl:none:0 8:bench_512:10:nccl:none:0 8:small:50:nccl:none:0" jobs/horeka_2node.slurm'
  ```

  Keep it if the 8-GPU rows move by more than the noise (the 512^3 row
  spread 0.3% across five runs of one night, the 256^3 row about 1%;
  the ghost-row line should drop by a quarter) and the 4-GPU rows stay a
  wash; the `small` row shows the fixed cost saved directly (3 x 0.2
  ms of its 2.5 ms of ghost rows).  If it is a wash, drop it and say so
  in FINDINGS: then the fixed cost is not per exchange but per byte
  stream, and the exchange chapter is closed.
- Docs: FINDINGS subsection with the table, README (the 2 x 4 A100
  column and the sentence on what remains), DESIGN.md departures, this
  file.

After (b) the remaining levers are (d), smaller records for the system
kinds whose matrix does not change between calls (bytes only, at most
3% at 512^3), and nothing cheap: the ghost rows' fixed cost needs
interior rows computed while the exchange is in flight.  Beyond the
exchange: a four-node run at 512^3 (expected well below 2x of two
nodes: the 13 ms of exchange stay while the compute halves), the H100
partitions (`accelerated-h100`, `cc90`), production runs, or merging the
branch into `main`.

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
branch, never the reverse; docs on main.  Task: lever (b) of
NEXT_SESSION.md, the two ghost exchanges at the end of linsolve as one
exchange of two fields (fill_ghosts(1, 3)), in the smallest form on
main and hst_mpi on the branch; results bit-identical; measured as an
A/B in one two-node job (time limit 12 min) and kept only with the
number; then docs (FINDINGS, README, DESIGN departures, NEXT_SESSION).
Safety net green after every code change (CPU, GPU, NCCL on istmcetus,
the cross-branch round trip).  Do not modify ~/Codes/hst/channel or
~/Codes/hst/hst-main.  Commit each step; push both branches at the end.
First thing: `! ssh horeka true` in the prompt.
```
