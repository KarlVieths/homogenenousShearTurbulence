# Next session: the y parallelisation without user input, then its exchange

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-28, session 8)

The y decomposition lives on the branch `multinode-y` and stays there:
the user's decision (2026-09-28) is to keep the two branches separate
for now, with the rule that **the same deck and the same field files
must work on both**.  `main` = x-z pencils, one rank owns all of y;
the branch adds `npy` y slabs, one per node, the ghost rows and the
line solver's reduced systems exchanged through a second NCCL
communicator over the y column (session 8, commit 54a3d2a).  Measured
on two HoreKA A100 nodes (FINDINGS.md, "The y decomposition", last two
subsections): 1.28x one node at 256^3 (0.0235 against 0.0301 s/step)
and 1.72x at 512^3 (0.1265 against 0.2169); with x-z pencils two nodes
were 2.3x *slower* than one.  What is left is the y exchange itself,
10.7 ms of the 23.5 ms step at 256^3 and 23.7 of 126.5 ms at 512^3,
mostly a fixed cost of about 0.5 ms per exchange (21 a step: 12 ghost
exchanges, 9 allgathers) between the nodes, not bandwidth.

The branch differs from `main` in `src/hst_mpi.f90` and
`src/hst_linsolve.f90` only (`git diff main multinode-y --stat`); all
fluid dynamics goes to `main` and reaches the branch by `git merge
main`, never the reverse; docs live on `main`.  HoreKA: `~/hst` =
`main`, `~/hst-y` = `~/hst-exp` = the branch head, all built.
`jobs/horeka_2node.slurm` takes `ROOTS="$HOME/hst-y $HOME/hst-exp"` to
run builds back to back in one two-node job (the only fair comparison,
NCCL phases vary 30% between jobs); a two-node job waited 6.5 h on
`accelerated`.  `ssh horeka` needs `! ssh horeka true` once in Claude
Code when the ControlMaster socket has expired; do not retry failed
logins.

## Compatibility of the two branches, as it is today

- *Field files*: identical layout (interior rows plus the file's four
  ghost rows, written by their owners on the branch, `hst_io`); both
  branches read the interior rows and refill the ghosts, so a restart
  file written by one runs on the other in either direction.  The
  branch's regressions already compare against `tests/reference/*.fld`
  written by `main`.  Not yet an explicit test: item 1 below.
- *Decks*: `npy` is in `&mesh` on both, default 1; `main` aborts on
  anything else ("npy > 1 is on the branch").  So a deck with `npy = 2`
  is a branch-only deck today: item 2 below removes that.
- *Launch*: each rank takes GPU `local_rank mod ndevices` from the
  launcher (`OMPI_COMM_WORLD_LOCAL_RANK` / Slurm, `hst_setup`);
  `transport = 'auto'` takes NCCL when there is one GPU per rank; the
  branch already counts the ranks per node
  (`MPI_Comm_split_type(..SHARED..)` in `setup_transport`).

## What to do

1. **A cross-branch round trip in the safety net** (`main`, a small
   script `tests/crossbranch.sh <build A> <build B> [npy]`, ~20
   lines): run `small` for 25 steps with A, restart with B for the
   other 25, compare with `tests/reference/small.fld` at 1e-10; on the
   ISTM boxes with A = the branch build at `npy = 2` and B = `main`,
   and the reverse.  The reference decks and scripts must not change.

2. **`npy = 0` = decide it yourself, the default on both branches**
   (the user's wish: on a multi-GPU, multi-node case the code, or at
   most a simple script, works out the layout).  The code does it,
   the scripts stay as they are:
   - `main`: `npy = 0` and `npy = 1` are the same thing (one line in
     `setup_decomposition`); `hst_input` default `npy = 0`; README.
   - branch: in `setup_decomposition`, before the split, count the
     ranks per node (move the `MPI_Comm_split_type` there; it is
     needed by `setup_transport` too, keep one call), `nnodes =
     nproc/node_ranks`; with `npy = 0` take `npy = nnodes` when it
     divides `ny` with `nyB >= 8` and `nproc/npy` divides `nx+1` and
     `nzd`, otherwise `npy = 1` with a one-line notice; require that a
     node's ranks are consecutive in `MPI_COMM_WORLD` (compare the
     node communicator's rank with `mod(iproc, node_ranks)`), which
     `mpirun --map-by ppr:N:node` and Slurm's block distribution give,
     and abort with a clear message if not.  The existing banner
     ("ranks = 8 (4 x-z pencils x 2 y slabs)") already says what was
     chosen.  About 30 lines in `hst_mpi`, nothing in the physics.
   - Then one deck runs unchanged on one GPU, on four GPUs of a node,
     and on two nodes, on either branch: the property to test, on
     istmcetus (one node, two GPUs: `npy` must come out 1 with two
     pencils) and on HoreKA in the two-node job (must come out 2).
   - Not to do: a Python launcher, a topology file, or an autotuner
     over layouts.  The job scripts keep their two Slurm lines
     (`--ntasks-per-node` = GPUs per node, `--map-by ppr:N:node`).

3. **The exchange, only with the number, one variant per two-node
   A/B** (rsync the variant to `~/hst-exp`, `make GPU=1 GPU_ARCH=cc80
   NCCL=1` there, then
   ```bash
   ssh horeka 'cd ~/hst-exp && sbatch --partition=accelerated --export=ALL,ROOTS="$HOME/hst-y $HOME/hst-exp",NPY_REG=2,CONFIGS="4:bench_256:20:nccl:none:1 8:bench_256:20:nccl:none:2 4:bench_512:10:nccl:none:1 8:bench_512:10:nccl:none:2 8:small:50:nccl:none:2" jobs/horeka_2node.slurm'
   ```
   keep a change only if the two-node step moves by more than the
   noise and the one-node rows stay a wash).  In the order of expected
   gain per line:
   - *First measure the fixed cost alone*: the `small` deck on 8 ranks
     with `npy = 2` and `timing` (the last CONFIGS entry above; add
     `tests/decks/small.in` handling to the job's `run` if it only
     looks in `examples/`): its exchanges carry almost no bytes, so
     the "transfers only" line is the per-exchange latency plus skew.
     If it is well below 0.5 ms per exchange, the bench numbers are
     skew between the slabs, and the levers below buy less than the
     bytes suggest.
   - (a) one NCCL group for the ghost rows instead of two (NCCL takes
     several sends to the same peer in a group, in order): one kernel
     per exchange, up to half of the 5.2 ms ghost-row time at 256^3.
     Five lines in `exchange_ghost_rows`.
   - (b) the two ghost exchanges that follow each other
     (`fill_ghosts(1)` then `fill_ghosts(3)` after the recovery of u
     and w, `hst_equations`) as one exchange of two fields: 12 to 9
     exchanges per step.  Needs a `fill_ghosts(c1, c2)` in
     `hst_derivatives` packing two components into one buffer: a
     small change in one physics file, so ask first.
   - (c) the allgather of one `line_chunk` batch overlapped with the
     forward sweep of the next (`chunk` is all the lines on the GPU
     today; two record buffers, NCCL on a second stream with events as
     the alltoall does): hides the bandwidth part, about 10 of the
     14.9 ms of records at 512^3, nothing of the fixed cost.
   - (d) smaller records for the kinds whose matrix does not change
     between calls (`KIND_DY` every substep; the implicit kinds while
     `deltat` is fixed): bytes only, 512^3 only.

4. **Docs**: FINDINGS.md (a subsection per measured item), README
   (the `npy = 0` default, the 2 x 4 A100 column), this file.

## Safety net (both branches; the branch adds a third argument, npy)

```bash
tests/run_tests.sh build-cpu 2         # 12 runs; on the branch also: tests/run_tests.sh build-cpu 4 2  and  ... 4 4
tests/run_tests.sh build-gpu 2         #                              tests/run_tests.sh build-gpu 2 2
tests/regression.sh build-cpu 2        # 50 steps of three decks at 1e-10; branch: tests/regression.sh build-cpu 4 2
tests/regression.sh build-gpu 1
tests/regression.sh build-gpu 2        #                                       branch: tests/regression.sh build-gpu 2 2
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && make GPU=1 NCCL=1 BUILD=build-nccl -j8 && make GPU=1 NCCL=1 BUILD=build-nccl test -j8 && tests/run_tests.sh build-nccl 2 2 && tests/regression.sh build-nccl 2 2 && tests/run_tests.sh build-nccl 2 && tests/regression.sh build-nccl 2'
```

The CPU build must be made in a shell that has *not* sourced
`env/istm.sh`, the GPU tests run in one that *has*; a flag change needs
the objects removed; `build-nccl` is exercised on istmcetus only, where
`2 2` (one pencil x two slabs) is the run that uses the NCCL y column
and `2` (two pencils) the NCCL alltoall.  A bit-for-bit comparison on
the CPU needs `FFTW_ESTIMATE` in `hst_fft`.  nvfortran 25.9 rejects the
names `kind` and `x` in OpenMP clauses of any file that can see
`hst_mpi` (FINDINGS.md).  Not to be done on `main`: the `buildrhs`
stencil-and-transpose tile and anything that adds a kernel for a few
percent (the user's call, session 5).

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (also ~/hst, ~/hst-y
and ~/hst-exp on HoreKA as rsync copies of main and of the branch
multinode-y), a GPU/CPU DNS for homogeneous shear turbulence; read
README.md, then NEXT_SESSION.md, then the last two subsections of
FINDINGS.md "The y decomposition".  The two branches stay separate and
must keep working with the same decks and field files.  Task, in this
order: the cross-branch round-trip test; npy = 0 as the default on both
branches meaning "the code chooses" (one slab per node on the branch,
from the ranks per node; no launcher scripts, the Slurm jobs stay as
they are); then the exchange levers of NEXT_SESSION.md, each measured
as an A/B in one two-node job and kept only with the number, the
fixed-cost measurement first.  Branches differ only in hst_mpi and
hst_linsolve; merge main into the branch, never the reverse; docs on
main.  Ask before any change that does not fit that (the two-field
ghost exchange touches a physics file).  Safety net green after every
code change (CPU, GPU, NCCL on istmcetus).  Do not modify
~/Codes/hst/channel or ~/Codes/hst/hst-main.  Commit each step; push
both branches at the end.  First thing: `! ssh horeka true` in the
prompt to open the HoreKA connection.
```
