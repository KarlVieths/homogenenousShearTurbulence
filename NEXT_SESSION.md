# Next session: measure the y decomposition on HoreKA (phase 3)

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-27, session 7)

The y decomposition is implemented and validated on the ISTM boxes, on
the branch `multinode-y` (FINDINGS.md, "The y decomposition"; DESIGN.md
7 (i) has the contract every physics file follows).  `main` has the
decomposition-agnostic physics (phase 0, bit-identical), `npy` in the
deck (must be 1 there), the `npy` argument of the test scripts and the
HoreKA jobs.  The branch differs from `main` in `src/hst_mpi.f90` and
`src/hst_linsolve.f90` only (`git diff main multinode-y --stat`); all
fluid dynamics goes to `main` and reaches the branch by `git merge
main`, never the reverse.

Validated (suite + three regression decks at 1e-10): CPU 2x1, 2x2, 1x4
slabs; RTX 3060 2x2; istmcetus NCCL 1x2 and 2x1; HoreKA one node 2x2
(suite, job 5167735) and the A/B at `npy = 1` (job 5167736, a wash:
FINDINGS.md).  Steps 1-3 below are therefore done; the two-node job
5167737 was queued on `accelerated` when the session ended (output
`~/hst-y/hst-2node-5167737.out`; if it never ran, resubmit as in step 4).  `ssh horeka` needs an interactive login when the ControlMaster
socket has expired: in Claude Code type `! ssh horeka true` once, the
master then persists 8 h; do not retry failed logins (HoreKA counts them).

## What to do

1. **Copy both trees to HoreKA and build.**  `~/hst` is `main`,
   `~/hst-y` the branch (never rebuild `~/hst` while a job is queued on
   it; `squeue -u xt8786` first):
   ```bash
   git checkout main       && rsync -a --delete --exclude 'build-*' --exclude .git --exclude 'hst-*.out' ./ horeka:hst/
   git checkout multinode-y && rsync -a --delete --exclude 'build-*' --exclude .git --exclude 'hst-*.out' ./ horeka:hst-y/
   ssh horeka 'for d in hst hst-y; do cd ~/$d && source env/horeka.sh gpu && make GPU=1 GPU_ARCH=cc80 NCCL=1 -j8 && make GPU=1 GPU_ARCH=cc80 NCCL=1 test -j8; done'
   ```
2. **Correctness on the A100 node** (dev_accelerated, one job): the
   suite and the small deck with 2 pencils x 2 slabs:
   ```bash
   ssh horeka 'cd ~/hst-y && sbatch --export=ALL,HST_ROOT=$HOME/hst-y,NPY=2 jobs/horeka_tests.slurm'
   ```
3. **The A/B at `npy = 1` must be a wash** (`main` against the branch
   on the same node, 1 and 4 GPUs, bench_256 and bench_512):
   ```bash
   ssh horeka 'cd ~/hst && sbatch --export=ALL,A_ROOT=$HOME/hst,B_ROOT=$HOME/hst-y jobs/horeka_ab.slurm'
   ```
   The branch at `npy = 1` skips the allgather and does the ghost rows
   in place; the two extra kernel launches per solve (the solve is split
   in two) are the only difference.  If B is slower by more than the
   30% run-to-run noise of the NCCL phase, look at the split first.
4. **The two-node measurement** (partition `accelerated`; a two-minute
   job waited 2-8 hours there in session 6; `dev_accelerated` rarely
   has two nodes free):
   ```bash
   ssh horeka 'cd ~/hst-y && sbatch --partition=accelerated --export=ALL,HST_ROOT=$HOME/hst-y,NPY_REG=2,CONFIGS="4:bench_256:20:nccl:none:1 4:bench_256:20:nccl:none:2 8:bench_256:20:nccl:none:2:info 8:bench_256:20:nccl:none:1 4:bench_512:10:nccl:none:1 4:bench_512:10:nccl:none:2 8:bench_512:10:nccl:none:2" jobs/horeka_2node.slurm'
   ```
   `CONFIGS` = ranks:deck:steps:transport:bind:npy[:info].  That gives,
   per deck: one node with x-z pencils (the reference, 0.030 / 0.216
   s/step in session 6), one node as 2 x 2 (the y path on NVLink), two
   nodes as 4 x 2 (the point of it all) and two nodes as 8 x 1 (the old
   2.1-2.3x slower case, for the record); at the end the regression
   decks on 8 ranks with `npy = 2`.  The timer prints the phases and,
   below the table, the pure transfer time of the ghost rows and of the
   allgathers ("of which y exchange").  Expected: two nodes at 256^3
   about 20 ms against 30 ms on one node (1.5x), y traffic about 3 ms
   (FINDINGS.md has the estimate and the istmcetus numbers).
5. **Write it up**: FINDINGS.md (a subsection under "The y
   decomposition"), README performance table (a "2 x 4 A100" column) and
   status row.  Docs live on `main` and are merged into the branch.
6. **Only if the timer says the allgather is a large exposed part of the
   step**, in this order, each measured with `jobs/horeka_ab.slurm`
   (A_ROOT = the branch copy, B_ROOT = the variant in `~/hst-exp`):
   `ncclAllGather` on `comm_y` in `allgather_y` (the NCCL communicator
   would be a second one, over the y column; ~30 lines in `hst_mpi`);
   the allgather of one `line_chunk` batch overlapped with the forward
   sweep of the next (`MPI_Iallgather`, two record buffers); a smaller
   record for the kinds whose matrix does not change (FINDINGS.md).  Do
   none of them without the number.

## Safety net (both branches; the branch adds a third argument, npy)

```bash
tests/run_tests.sh build-cpu 2         # 12 runs; on the branch also: tests/run_tests.sh build-cpu 4 2  and  ... 4 4
tests/run_tests.sh build-gpu 2         #                              tests/run_tests.sh build-gpu 2 2
tests/regression.sh build-cpu 2        # 50 steps of three decks at 1e-10; branch: tests/regression.sh build-cpu 4 2
tests/regression.sh build-gpu 1
tests/regression.sh build-gpu 2        #                                       branch: tests/regression.sh build-gpu 2 2
ssh istmcetus 'cd ~/Codes/hst/homogenenousShearTurbulence && source env/istm.sh && make GPU=1 NCCL=1 BUILD=build-nccl -j8 && make GPU=1 NCCL=1 BUILD=build-nccl test -j8 && tests/run_tests.sh build-nccl 2 2 && tests/regression.sh build-nccl 2 2'
```

The CPU build must be made in a shell that has *not* sourced
`env/istm.sh`, the GPU tests run in one that *has*; a flag change needs
the objects removed; `build-nccl` is exercised on istmcetus only.  A
bit-for-bit comparison on the CPU needs `FFTW_ESTIMATE` in `hst_fft`
(FFTW_MEASURE plans differ run to run at 5e-15).  nvfortran 25.9 rejects
the names `kind` and `x` in OpenMP clauses of any file that can see
`hst_mpi` (FINDINGS.md).

## Where the one-node time goes (FINDINGS.md), for reference

One A100, bench_256: cuFFT 31% of the kernel time, `buildrhs` 17%, line
solver 14% (its two sweeps each at two thirds of the bandwidth; what is
left is the strided read of the y-innermost field, structural), tiled
transpose 10%, `build_products` 10%.  Four A100: the exposed alltoall
about 13% of the step.  Not to be done on `main`: the `buildrhs`
stencil-and-transpose tile (the user's call, notes in FINDINGS.md of
session 5) and anything that adds a kernel for a few percent.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (also ~/hst and
~/hst-y on HoreKA as rsync copies of main and of the branch multinode-y),
a GPU/CPU DNS for homogeneous shear turbulence; read README.md, then
NEXT_SESSION.md (phase 3 of the y decomposition: the HoreKA
measurement), then FINDINGS.md "The y decomposition".  Task: the work
plan of NEXT_SESSION.md: build both trees on HoreKA, the correctness job
at npy = 2, the A/B at npy = 1 (must be a wash), the two-node job on
`accelerated`, then FINDINGS/README.  Only touch the exchange (NCCL
allgather, overlap) if the timer says it is a large exposed part of the
step, and measure each change.  Branches differ only in hst_mpi and
hst_linsolve; merge main into the branch, never the reverse; docs on
main.  Ask before any change that does not fit that.  Safety net green
after every code change (CPU, GPU, NCCL on istmcetus).  Do not modify
~/Codes/hst/channel or ~/Codes/hst/hst-main.  Commit each step; push
both branches at the end.  First thing: `! ssh horeka true` in the
prompt to open the HoreKA connection.
```
