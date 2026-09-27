# Next session: the y decomposition, on the branch `multinode-y`

Copy the block at the end as the opening message of the next session.

## Why, and what the measurements say

With x-z pencils every rank owns all of y and each of the 27 field
transposes per step is an alltoall over all ranks.  On one node that
runs on NVLink and is 13% of the step (exposed).  On two nodes half of
every alltoall crosses the node's single HDR InfiniBand adapter (25 GB/s
each way), 1.38 GB per node per step at 256^3, and 8 A100 are 2.1-2.3x
*slower* than 4 (FINDINGS.md, "The second node").  GPUDirect RDMA, NUMA
binding of the ranks and a two-level alltoall were measured and do not
help: NCCL is at the wire rate and the volume is fixed by the
decomposition.  The only lever is to change what crosses the node: keep
the x-z alltoalls inside a node and split y across nodes, because y is
the one direction where the coupling is local (five-point compact
stencils and line solves that border well).

Expected traffic across nodes with y slabs at 256^3, 2 nodes x 4 GPUs
(npxz = 4, npy = 2): the velocity ghost rows, about five fields per
substep x four planes x 131 KB, plus the reduced systems of the four
line solves, 20 complex numbers per line = 2.6 MB per solve and rank:
about 40 MB per rank per step against 1380 MB per node now.  Over MPI
on device buffers (15 GB/s on HoreKA) that is under 3 ms of a step that
halves from 30 to about 15 ms of compute, so the two-node step should
come out near 18-20 ms, 1.5-1.7x the one-node step, and four nodes at
512^3 similarly.  The products need no exchange at all: they are formed
on the ghost rows too (`assemble_vvdz` copies rows ny0-2..nyN+2), so
the image rows of the products come out of the transforms for free, as
they do today.

## The two branches

`main` stays the simple code: `npy = 1`, every rank owns all of y.
`multinode-y` carries the y decomposition and nothing else.  All fluid
dynamics lives on `main` and reaches the branch by `git merge main`
(never cherry-picks, never the other direction), so the two differ only
in the parallel layer.  The way to make that true and keep it true:

- **The parallel layer is three files**: `hst_mpi.f90` (decomposition,
  communicators, transposes, ghost-row exchange, the allgather of the
  reduced systems, the MPI-IO views), `hst_linsolve.f90` (the reduced
  interface system) and, if it cannot be avoided, `hst_io.f90` (which
  rows a rank writes).  Every other file must be identical on both
  branches.  Check with `git diff main multinode-y --stat` after every
  merge: only those files may appear.
- **The contract every physics file follows** (write it into DESIGN.md
  in phase 0, section 7 (i), so that new features on `main` respect it
  without thinking about the branch):
  1. y loops run `ny0, nyN`, never `0, ny - 1`; a stencil along y reads
     rows `iy + j` for `j = -2..2`, which are the two ghost rows on each
     side, and nothing further away.
  2. Ghost rows are filled only by `fill_ghosts` / `fill_ghosts_field`
     (`hst_derivatives`), which call one routine of `hst_mpi`,
     `exchange_ghost_rows(field, ph)`; on `main` that routine is the
     wrap with the phase that `fill_ghosts_field` does today.
  3. A box integral is a partial sum over the rank's rows followed by
     `MPI_Allreduce` over `MPI_COMM_WORLD` (as `hst_stats` does); a rank
     that has no share contributes zero (the pressure mean of the (0,0)
     mode, `hst_pressure`, is the one place written differently today).
  4. A row with special treatment (the Stokes profile at rows 0 and
     ny-1, `hst_stokes`) is touched only if `ny0 <= iy <= nyN`.
  5. Anything else along y goes through `line_solve` (one entry point,
     one signature, on both branches).
  6. Global y arrays (`y`, `dyl`, `fy`, `inlayer`, `der`) stay global,
     indexed by the global row: they are small and every rank has them.
  With that, `npy = 1` on the branch must give results bit-identical to
  `main`, and `npy > 1` must reproduce `tests/reference/*.fld` at 1e-10
  (the initial field is seeded by global indices, so any decomposition
  starts from the same field).

## The scheme (the simplest one that is right)

One level, no tree, no autotuner: the channel's `ys_solve_endpoint_schur`
plus `y_schur_solver.f90` do the same thing in 2000 lines with a
multi-level reduction and an autotuned pass hierarchy
(`channel/src/linsolve/`, see the notes at the end); with `npy <= 8`
the reduced system per line has at most 16 unknowns and every rank of a
y column solves it redundantly.

**Decomposition.**  Process grid `npxz x npy`, `nproc = npxz*npy`,
`ipy = iproc/npxz`, `ipxz = mod(iproc, npxz)`, so that the `npxz` ranks
of a y slab are consecutive and one node holds one slab
(`mpirun --map-by ppr:4:node`).  `npy` is a deck parameter in `&mesh`
(default 1); `npxz` must divide `nx+1` and `nzd` as now, `npy` must
divide `ny` with `nyB = ny/npy >= 8` (the sweep's rows 0, 1, m-2, m-1
must be four different rows).  `ny0 = ipy*nyB`, `nyN = ny0 + nyB - 1`;
the arrays are already `(ny0-2:nyN+2, ...)`.  Two communicators:
`comm_xz` (constant `ipy`: the transposes, MPI or NCCL with one NCCL
communicator per slab, its unique id broadcast inside `comm_xz`) and
`comm_y` (constant `ipxz`: the ghost rows and the reduced systems).
`has_average` stays `nx0 == 0`.

**Ghost rows** (`exchange_ghost_rows` in `hst_mpi`): send rows
`nyN-1:nyN` to `ipy+1` and rows `ny0:ny0+1` to `ipy-1` (both modulo
`npy`), receive into `ny0-2:ny0-1` and `nyN+1:nyN+2`; the rank with
`ipy = 0` multiplies what it receives from below by `conjg(ph)`, the
rank with `ipy = npy-1` what it receives from above by `ph` (the
shear-periodic images, exactly the four lines of today's
`fill_ghosts_field`, which is the `npy = 1` case: sender = receiver).
A plane is strided in memory (iy innermost), so pack and unpack are two
small kernels; the transfer is `MPI_Sendrecv` on device buffers inside
`use_device_addr` after a `cudaStreamSynchronize` (the pattern of the
MPI transport in `alltoall_start`).  Start with MPI for all y traffic:
it is a few MB per step; switch the two calls to NCCL only if the timer
says so.

**The line solve** (`cyclic_penta_solve` generalised; today's code is
the `npy = 1` case, keep its structure and names).  Per line, segment s
(this rank's rows) has `m = nyB - 2` interior rows and its border
`b_s = (x(nyN-1), x(nyN))`.  Interior rows 0 and 1 have entries that
point at the previous segment's border `b_{s-1}` (today: the wrapped
entries, with `conjg(ph)`; on the branch the phase only for `s = 0`);
rows m-2 and m-1 have entries that point at `b_s` (today: kept in
`U1`, `U2`, seeding the back substitution).  The forward sweep is
unchanged.  What changes is that the coefficients of `b_{s-1}` and of
`b_s` must be kept apart instead of being added into one `pm`, `qm`,
`t0p`, `t0q` (they are the same unknown only when `npy = 1`): after the
sweep the rank knows rows 0, 1, m-2, m-1 of its segment as
`x = xm - p1 b_{s-1,1} - p2 b_{s-1,2} - q1 b_{s,1} - q2 b_{s,2}`, five
complex numbers per row, **20 per line**.  The border rows of segment s
(rows nyN-1, nyN) couple to its own rows m-2, m-1 (`c1m4`, `c1m3`,
`c2m3`), to `b_s` (`d11`..`d22`) and to rows 0, 1 of segment s+1
(`c10`, `c20`, `c21`, with `ph` only for `s = npy-1`); all of these
come from `der` and `k2`, which every rank has, so no coefficients need
exchanging.  Substituting the four rows gives the block row
`A_s b_{s-1} + D_s b_s + C_s b_{s+1} = r_s` (2x2 blocks): a
block-tridiagonal cyclic system of `2 npy` unknowns per line.

So the kernel splits in two around one communication:
1. forward sweep + the 20 numbers per line into a buffer
   `reduced(20, lines)`;
2. `MPI_Allgather` of that buffer over `comm_y` (device buffers), one
   call per `line_chunk` batch;
3. every rank assembles the `2npy x 2npy` system of each of its lines
   from the `npy` rows it now has, solves it by Gaussian elimination
   without pivoting in thread-private storage (16x16 complex at most; the
   2x2 Schur solve of today is this at `npy = 1`), takes `b_{s-1}` and
   `b_s`, and does the backward sweep as today with `yb = conjg(ph) *
   b_{s-1}` only for `s = 0` and the seed `b_s`.
The workspace `U1, U2, B1, B2, X` stays between the two kernels (module
arrays).  Cost: about 1400 complex flops per line for the dense solve at
`npy = 8`, nothing against the sweeps.  Do not exploit the block
structure and do not pivot unless `test_linsolve` says so (the systems
are diagonally dominant).

**I/O.**  The CPL file has rows -2..ny+1; each rank writes its rows
`ny0..nyN` through a subarray view with a y offset, `ipy = 0` also the
file rows `ny, ny+1` and `ipy = npy-1` the rows `-2, -1` (the file's
ghost rows are plain copies, as `hst_io` writes them today with
`modulo(iy, ny)`).  Reading: the interior rows, then `fill_ghosts`, as
now.  Snapshots and the pressure the same.

**Everything else** is untouched: transforms and products (fewer y
planes per rank, the same code on `comm_xz`), `shear_shift` (per-row
phase), the initial field (seeded by global index), statistics
(partial sums + allreduce), CFL (allreduce max), the tests.

## Work plan, each step committed and tested

0. **On `main`: make the physics files decomposition-agnostic** (a
   no-op there, so every regression stays bit-identical): the contract
   above, i.e. `ny0, nyN` in `hst_equations`, `hst_pressure`,
   `hst_stats`, `hst_stokes`, `hst_derivatives`, `hst_setup`, `hst_io`
   (grep `0, ny - 1` and `ny - 1`, `ny - 2`); `exchange_ghost_rows` in
   `hst_mpi` holding today's wrap, called from `fill_ghosts_field`; the
   pressure mean as partial sum + allreduce; the Stokes rows by
   ownership; the file copy over the rows the rank owns; the contract
   in DESIGN.md 7 (i).  Run the whole safety net (CPU, GPU one and two
   ranks, NCCL on istmcetus).  Then `git checkout multinode-y && git
   merge main`.
1. **Branch: decomposition and ghost rows.**  `npy` in `&mesh` and
   `hst_params`; `setup_decomposition` with the two communicators; the
   transposes on `comm_xz` (MPI and NCCL); `exchange_ghost_rows` with
   the pack/unpack kernels and `MPI_Sendrecv`; the I/O views.  Test:
   `test_roundtrip` (transforms and restart file, no line solve) on 2
   and 4 CPU ranks with `npy = 2`; then `npy = 1` on 2 ranks
   bit-identical to `main`.
2. **Branch: the line solve.**  The 20-number pack, the allgather, the
   reduced solve, the backward sweep with the two borders.  Test:
   `test_linsolve` with `npy = 2` (the host operator through the ghost
   rows is already decomposition-agnostic), then the full suite and the
   regression decks with `npy = 2` on 2 and 4 CPU ranks (`small` has
   `ny = 32`: `nyB = 16`), then GPU: istmcetus with 2 ranks `npy = 2`
   (`npxz = 1`, so the transposes are local and only the y path is
   exercised), then HoreKA one node `npxz = 2, npy = 2` against
   `npxz = 4` (same node, `jobs/horeka_ab.slurm` with two decks), then
   two nodes `npxz = 4, npy = 2` (`jobs/horeka_2node.slurm`, add the
   deck's `npy` to its `CONFIGS`).  Extend `tests/run_tests.sh` and
   `tests/regression.sh` with an optional `npy` argument (a `sed` into
   the deck, as the jobs do) so that the branch's safety net is one
   command per decomposition.
3. **Measure and finish.**  Timer phases for the y exchange and the
   reduced solve (`hst_timer`: two new phases on the branch only);
   FINDINGS.md section; README table with a "2 nodes" column; the
   NCCL switch for the y traffic if the exchange is more than a few
   percent of the step.  The A/B against `main` on one node at
   `npy = 1` must be a wash (the two kernels and the allgather cost
   nothing when `npy = 1`: skip the allgather there).

Expected size of the branch's difference to `main`: about 250 lines in
`hst_mpi` (communicators, exchange with two kernels, allgather, views),
120 in `hst_linsolve` (the pack, the reduced solve, the split), a few
in `hst_io`.  If it grows well beyond that, stop and ask.

## Safety net (both branches)

```bash
tests/run_tests.sh build-cpu 2       # 12 runs: transforms, solver kinds, Kelvin (5 decks),
tests/run_tests.sh build-gpu 2       #   pressure, Taylor-Green, forcing, conservation, Stokes
tests/regression.sh build-cpu 2      # 50 steps of three decks against tests/reference/*.fld at 1e-10
tests/regression.sh build-gpu 1
tests/regression.sh build-gpu 2      # the MPI transport on one GPU (two ranks)
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && tests/run_tests.sh build-nccl 2 && tests/regression.sh build-nccl 2'
```

The CPU build must be made in a shell that has *not* sourced
`env/istm.sh`, the GPU tests run in one that *has*; a flag change needs
the objects removed; `build-nccl` is exercised on istmcetus only.  On
HoreKA `~/hst` is an rsync copy (never rebuild it while a job on it is
queued; variants in `~/hst-exp*`):

```bash
rsync -a --delete --exclude 'build-*' --exclude .git --exclude 'hst-*.out' ./ horeka:hst/
ssh horeka 'cd hst && source env/horeka.sh gpu && make GPU=1 GPU_ARCH=cc80 NCCL=1 -j8 && make GPU=1 GPU_ARCH=cc80 NCCL=1 test'
sbatch jobs/horeka_tests.slurm        # suite on 4 A100 (NCCL) + 4-GPU run against a CPU run
sbatch jobs/horeka_bench_all.slurm    # bench_64/256/512 on 1 and 4 A100 (the README table)
sbatch --export=ALL,A_ROOT=$HOME/hst,B_ROOT=$HOME/hst-exp jobs/horeka_ab.slurm       # A/B: both builds, same node
sbatch --partition=accelerated jobs/horeka_2node.slurm        # 4 A100 on one node against 8 on two; CONFIGS and NCCL_* overridable
sbatch --export=ALL,HST_ROOT=$HOME/hst-exp,KERNEL=regex:linsolve jobs/horeka_ncu.slurm   # Nsight Compute counters
```

`dev_accelerated` has three nodes and rarely two free: two-node jobs go
to `accelerated` (a two-minute job waited 2-8 hours there).  The
hardware counters are open on the compute nodes (`jobs/horeka_ncu.slurm`).

## Where the one-node time goes (FINDINGS.md), for reference

One A100, bench_256: cuFFT 31% of the kernel time, `buildrhs` 17%, line
solver 14% (its two sweeps each at two thirds of the bandwidth; what is
left is the strided read of the y-innermost field, structural), tiled
transpose 10%, `build_products` 10%.  Four A100: the exposed alltoall
about 13% of the step.  Not to be done on `main`: the `buildrhs`
stencil-and-transpose tile (the user's call, notes in FINDINGS.md of
session 5 and in the git history of this file) and anything that adds
a kernel for a few percent.  Step times: README table.

## Notes on the channel code's y machinery (read, not copied)

`channel/src/linsolve/y_line_solvers.fypp:1101-1469`
(`ys_solve_endpoint_schur`: two exposed rows per interface, the
interior eliminated against five right-hand sides, 20 complex numbers
per line packed) and `y_schur_solver.f90` (958 lines: a tree of levels
with arity 4/3/2 from `pass_node_counts`, alltoall or allgather per
level, the root solved redundantly), plus a pipelined-LU alternative
(`:1471-1924`), halo exchange (`:1925-2065`, `MPI_Isend/Irecv` on
`MPI_COMM_Y`, CUDA-aware) and the autotuner
(`src/mpi/mpi_autotune.f90`).  All of it is for an open interval in y;
the cyclic case adds the corner blocks of the reduced system, which is
why ours is written from the bordering rather than adapted from there.
The 20-per-line width is the same in both, which is a check on the
count above.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (also ~/hst on HoreKA as
an rsync copy), a GPU/CPU DNS for homogeneous shear turbulence; read
README.md, then NEXT_SESSION.md (the y-decomposition handout), then
FINDINGS.md "The second node".  Task: the work plan of NEXT_SESSION.md:
phase 0 on main (the decomposition-agnostic refactor, bit-identical),
then phases 1-2 on the branch multinode-y (npy in the deck, ghost-row
exchange, the reduced interface system of the line solve), validated
with npy = 2 on CPU ranks and on istmcetus before any HoreKA job; phase
3 (two-node numbers) if there is time.  Simplest solution, readable
code, the branches differ only in hst_mpi, hst_linsolve and hst_io;
merge main into the branch, never the reverse.  Ask before any change
that does not fit that.  Safety net green after every step (CPU, GPU,
NCCL on istmcetus).  Do not modify ~/Codes/hst/channel or
~/Codes/hst/hst-main.  Commit each step; push both branches at the end.
```
