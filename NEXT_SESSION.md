# Next session: the y decomposition (WP6), if the user wants a second node

Copy the block at the end as the opening message of the next session.

State at the end of this one: `main` at the commit that adds this file,
pushed.  Item 1 of the previous handoff is done and closed: the line
solver's interior matrix turned out to be real (the wrap phase only
touches the border), so the factor and the border columns are real,
128 bytes per row instead of 192 and one real division per row; the
backward sweep loads four rows before using any (the compiler put each
row's loads behind the previous row's stores).  The sweep kernel went
from 1834 to 1219 us per call at 256^3 on the A100 (its two sweeps are
each at two thirds of the bandwidth; the remaining stalls are the
strided reads of the y-innermost field, which the one-thread-per-line
structure cannot avoid), the step by 5% on one A100 and 11% on four at
256^3, 6% and 5% at 512^3, and the RTX 3060 and the CPU by 14-21%.  The
register-idea list of the old handoff is moot: the kernel fits in 80
registers and lower caps gain nothing (FINDINGS.md, session 6).  Read
`README.md`, then FINDINGS.md (the last section), then this file.

Item 2, the two-node measurement, is done (FINDINGS.md, last section):
**8 A100 on two nodes are 2.1-2.3x slower than 4 on one** (bench_256
0.074 against 0.034, bench_512 0.52 against 0.23; MPI transport 3.8x
worse still).  The solves halve, the transposes explode: with x-z
pencils half of every alltoall crosses InfiniBand, and even at the full
rate of the node's links that share alone would exceed half of the
one-node step.  Deeper overlap cannot hide it.  So the code stops at
one node until it has the y decomposition (WP6, DESIGN.md 7 (i)), the
only item left that adds real structure.  `~/hst` on HoreKA is rebuilt
from this `main`; the leftover dev-partition copy of the two-node job
was cancelled.

## Safety net, use it before and after every change

```bash
tests/run_tests.sh build-cpu 2       # 12 runs: transforms, solver kinds, Kelvin (5 decks),
tests/run_tests.sh build-gpu 2       #   pressure, Taylor-Green, forcing, conservation, Stokes
tests/regression.sh build-cpu 2      # 50 steps of three decks against tests/reference/*.fld at 1e-10
tests/regression.sh build-gpu 1
tests/regression.sh build-gpu 2      # the MPI transport on one GPU (two ranks)
```

The CPU build must be made in a shell that has *not* sourced
`env/istm.sh` (that puts NVHPC's mpifort first on PATH), and the GPU
tests must be run in one that *has* (the system `mpirun` cannot launch
the NVHPC binary; the symptom is "run failed" for every deck).  A change
to the Makefile's flags does not recompile the objects: `rm` the object
(or the build directory) first.  An NCCL build on the ISTM boxes
(`make GPU=1 NCCL=1 BUILD=build-nccl`) can only be exercised on istmcetus
(two RTX A6000; the second stream of the overlap is only used there and on
HoreKA):

```bash
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && tests/run_tests.sh build-nccl 2 && tests/regression.sh build-nccl 2'
```

On HoreKA (`~/hst` there is an rsync copy of the repository, not a
checkout; never rebuild it while a job on it is queued; A/B variants go
in `~/hst-exp`, `~/hst-exp2`, ... which are rsync copies too):

```bash
rsync -a --delete --exclude 'build-*' --exclude .git --exclude 'hst-*.out' ./ horeka:hst/
ssh horeka 'cd hst && source env/horeka.sh gpu && make GPU=1 GPU_ARCH=cc80 NCCL=1 -j8 && make GPU=1 GPU_ARCH=cc80 NCCL=1 test'
sbatch jobs/horeka_tests.slurm        # suite on 4 A100 (NCCL) + 4-GPU run against a CPU run
sbatch jobs/horeka_bench_all.slurm    # bench_64/256/512 on 1 and 4 A100 (the README table)
sbatch jobs/horeka_profile.slurm      # phase timer + nsys kernel summary on 1 A100
sbatch --ntasks-per-node=4 --gres=gpu:4 --export=ALL,NP=4 jobs/horeka_profile.slurm   # 4 GPUs, mpi and nccl
sbatch --export=ALL,A_ROOT=$HOME/hst,B_ROOT=$HOME/hst-exp jobs/horeka_ab.slurm       # A/B: both builds, same node, timer + nsys
sbatch --export=ALL,A_ROOT=$HOME/hst-exp,B_ROOT=$HOME/hst-exp2,C_ROOT=$HOME/hst-exp3 jobs/horeka_ab.slurm   # three-way
sbatch --export=ALL,HST_ROOT=$HOME/hst-exp,KERNEL=regex:buildrhs,SKIP=6,COUNT=2 jobs/horeka_ncu.slurm   # Nsight Compute counters
sbatch --partition=accelerated jobs/horeka_2node.slurm        # 4 A100 on one node against 8 on two (dev has 3 nodes: use accelerated); CONFIGS and NCCL_* overridable
```

`dev_accelerated` runs one job per user at a time and queues at most
four; a two-minute job waits 5-40 minutes, and it has only three nodes,
so a two-node job may sit there for hours ("nodes required are down,
drained or reserved"): submit it on `accelerated` as well.  The A/B job
is the way to time a change: the same node, both builds back to back,
and NCCL's alltoall varies by 30% between runs, so never compare
alltoall phases across jobs.  The hardware counters are open on the
HoreKA compute nodes (`jobs/horeka_ncu.slurm`: DRAM bytes, achieved
occupancy, registers, stall reasons of a kernel family), not on the ISTM
boxes.  A diagnostic build with one part of a kernel removed, timed
with the ncu job, is the cheap way to split a kernel's time (session 6
did it for the two sweeps of the solver).  The RTX 3060 is good for
kernel-level A/Bs with nsys and for reading the SASS
(`cuobjdump -sass build-gpu/<file>.o`: the order of loads and stores in
a loop is what decided session 6's backward sweep), but it runs FP64 at
1/64 rate, so it does not predict the A100.

## Where the time goes now (FINDINGS.md, last section)

One A100, bench_256, kernel time: cuFFT 31% (four transforms),
`buildrhs` 17%, line solver 14% (1.22 ms per call for 1.08 GB, its two
sweeps each at about two thirds of the bandwidth), tiled transpose 10%,
`build_products` 10%, `buildrhs_prepare` 5%, `assemble_vvdz` 4%.  Four
A100: the timer phases are "to physical" 23% and "products" 56% (the
transposes have no phase of their own, only their exposed part counts),
the two solve phases 14%; the exposed alltoall is about 13% of the step.

## What is worth doing, in order

1. **The y decomposition (WP6), only if the user asks for a second
   node.**  DESIGN.md 7 (i): a 2D decomposition npx x npy with each
   node holding one y slab (npy = number of nodes), so that the x-z
   alltoalls stay on NVLink inside a node and only the line solver's
   border couplings (the Schur complement of the bordering, of which
   the present solve is the npy = 1 case: `hst_linsolve` already says
   so) and the ghost rows cross the node boundary.  It touches the
   decomposition (`hst_mpi`), the ghost rows (`hst_derivatives`), the
   line solver, the I/O types and the statistics; it is the largest
   change since the start and must be designed with the user first.
   If the user does not need more than four A100 per run, skip it:
   the code is done at one node.  What cannot help is already measured
   (FINDINGS.md, the second node): GPUDirect RDMA, NUMA binding of the
   ranks (`jobs/bind_numa.sh`) and a two-level alltoall (commit c151db4,
   reverted); the flat NCCL alltoall runs at the wire rate of the
   node's single HDR adapter.
2. **Deeper overlap** (the alltoalls of the first product group behind
   the products and `buildrhs` of the second; six products in memory at
   once): at most the exposed 13% of the 4-GPU step on one node, more
   likely half of it.  Worth it only for one-node runs, and only if
   the user wants those last percent.
3. **Memory per rank** at 512^3: 19.5 GB of transform buffers on one
   rank, 4.9 GB on four; the line-solver workspace with all columns is
   48 B x lines x ny (`line_chunk` bounds it); the transpose buffers
   are two pairs of one field each.
4. **The line solver, if ever again**: what is left is structural.  Its
   stalls are the strided reads of the y-innermost field (one line per
   thread, 32 sectors per warp request; the KIND_DY solve reads it five
   times per row through its stencil, a sliding window would recover
   about 1% of the step) and the write-through of the workspace.
   Prefetching the next row's right-hand side or stencil coefficients
   was measured and gained nothing (FINDINGS.md).  A coalesced layout
   would need a gather pass (the bytes session 5 removed) or a
   different thread mapping; neither is local.

**Not doing: `buildrhs` as a stencil-and-transpose tile** (decided
2026-09-26, the user's call: the code should stay as simple as it is).
The notes, in case it is ever needed.  `buildrhs` is 17% of the 1-GPU
kernel time, 2.4 ms per call at 256^3 at about half the bandwidth: its
fifteen `VVdz` stencil reads per product are a plane apart between
neighbouring threads (iy innermost, `VVdz` has iz innermost), so half of
every 32-byte sector is wasted; the other loop order (iz innermost, `rhs`
strided) was tried in session 3 and lost, because with iy innermost the
five-point windows of neighbouring threads overlap and L1 serves most of
the reads.  The fix would be a tile through shared memory in the pattern
of `transpose_tiled`: read a (iz, iy) tile of `VVdz` with two ghost rows
on each side along iy, apply the three stencils (D0, D1, D2) along iy in
the tile, write the result with iy contiguous.  Two forms: (a) the
kernel also does the six product cases, the mean-mode packing and the
Stokes `no_mean_vw` rule, i.e. the physics of `buildrhs` moves into a
CUDA Fortran kernel while the CPU keeps the OpenMP version (two copies
of the equations, and DESIGN.md 7's rule that CUDA Fortran stays in
hst_mpi and hst_fft would need extending); or (b) the kernel only writes
the three derivatives of each product into a temporary in the `rhs`
layout and the present loop reads them contiguously (one copy of the
physics, but nine field writes and reads per group added, which eats
part of the gain).  About 100 lines either way.  Ceiling 7% of the
1-GPU step at full bandwidth, realistically 3-4% for form (b).
`buildrhs_prepare` (5%) could be folded into it (costed at 2% of the
step, session 4).

Things learned this session that the next one should not relearn
(FINDINGS.md has the numbers): the interior matrix of the cyclic solve
is real (only the border carries the phase); a scalar `p1` clashes with
an array `P1` (case-insensitive names, like `u1`/`U1` last time);
`tests/regression.sh` used to test the exit status of `tail` instead of
the comparison and printed REGRESSION OK on a failure (fixed: the
`PIPESTATUS`); nvfortran unrolls a loop but does not hoist the loads of
one iteration above the stores of the previous one, so write the loads
of several rows first when a sweep is latency-bound; `cuobjdump
-res-usage` shows the registers a cap leaves, not what the kernel needs
(compile with lower caps until LOCAL becomes nonzero); a 255-block grid
on 108 SMs cannot use occupancy, only per-thread memory parallelism.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (also ~/hst on HoreKA as
an rsync copy), a GPU/CPU DNS for homogeneous shear turbulence; read
README.md, FINDINGS.md (last section) and NEXT_SESSION.md.  Task:
NEXT_SESSION.md: the two-node measurement showed that x-z pencils stop at
one node, so the only item left is the y decomposition (WP6), which adds
real structure; discuss with me whether to do it before touching code.
The code must stay as simple as it is
(no new kernels; the buildrhs tile is documented and not to be done; the
line solver is finished).  tests/run_tests.sh and tests/regression.sh
green after every step (CPU, GPU, and the NCCL build on istmcetus),
HoreKA jobs for the numbers.  Do not modify ~/Codes/hst/channel or
~/Codes/hst/hst-main.  Commit each step; push at the end.
```
