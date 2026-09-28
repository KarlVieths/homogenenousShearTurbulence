# Next session: what to do with the y decomposition now that it is measured

Copy the block at the end as the opening message of the next session.

## Where things stand (end of 2026-09-28, session 8)

The y decomposition (branch `multinode-y`) is measured on two HoreKA
A100 nodes (FINDINGS.md, "The y decomposition", the last two
subsections).  With one slab per node and the ghost rows and the line
solver's reduced systems exchanged through a second NCCL communicator
over the y column (commit 54a3d2a, the session's one code change, in
`src/hst_mpi.f90` only), two nodes are 1.28x one node at 256^3 (0.0235
against 0.0301 s/step) and 1.72x at 512^3 (0.1265 against 0.2169).
With x-z pencils on two nodes the step was 2.3x *slower* than on one.
Regression decks on 8 ranks with `npy = 2` at 5e-14; the branch at
`npy = 1` and everything on one node are unchanged (A/B job 5168032).

What is left is the y exchange itself: 10.7 ms of the 23.5 ms step at
256^3, 23.7 of 126.5 ms at 512^3, mostly a fixed cost of about 0.5 ms
per exchange (21 a step) between the two nodes, not bandwidth.  The
branch still differs from `main` in `src/hst_mpi.f90` and
`src/hst_linsolve.f90` only; all fluid dynamics goes to `main` and
reaches the branch by `git merge main`, never the reverse; docs live on
`main`.  HoreKA: `~/hst` = `main`, `~/hst-y` = the branch (rebuilt at
the end of session 8), `~/hst-exp` = the same commit (the A/B variant).
`jobs/horeka_2node.slurm` takes `ROOTS="$HOME/hst-y $HOME/hst-exp"` to
run several builds back to back in one two-node job (the only fair
comparison: NCCL phases vary 30% between jobs); a two-node job waited
6.5 h on `accelerated` this time.  `ssh horeka` needs `! ssh horeka
true` once in Claude Code when the ControlMaster socket has expired; do
not retry failed logins.

## What to do

Two decisions are the user's, not the session's; ask first:

1. **Whether to squeeze the exchange further.**  FINDINGS.md lists the
   levers with their expected gain: (1) one NCCL group for the ghost
   rows instead of two (five lines in `exchange_ghost_rows`; worth up
   to half of the 5.2 ms ghost-row time at 256^3 if the cost is per
   kernel), (2) the allgather of one `line_chunk` batch overlapped with
   the forward sweep of the next (a second stream and events, as the
   alltoall does; hides bandwidth, about 10 of the 14.9 ms of records
   at 512^3, nothing at 256^3), (3) smaller records for constant-matrix
   kinds (bytes only, 512^3 only).  Each is an rsync to `~/hst-exp`,
   `make GPU=1 GPU_ARCH=cc80 NCCL=1` there, and
   ```bash
   ssh horeka 'cd ~/hst-exp && sbatch --partition=accelerated --export=ALL,ROOTS="$HOME/hst-y $HOME/hst-exp",NPY_REG=2,CONFIGS="4:bench_256:20:nccl:none:1 8:bench_256:20:nccl:none:2 4:bench_512:10:nccl:none:1 8:bench_512:10:nccl:none:2" jobs/horeka_2node.slurm'
   ```
   Keep a change only if the two-node step moves by more than the
   run-to-run noise (compare A and B of the same job; the one-node
   rows must stay a wash).
2. **Whether the branch becomes `main`.**  It is validated everywhere
   `main` is, and at `npy = 1` it costs 0.5-1.3% of the step (the solve
   split in two, FINDINGS.md "HoreKA, one A100 node"); the price is
   +230 lines in `hst_linsolve` and +230 in `hst_mpi` that a reader of
   the one-node code has to see.  If merged: `git merge multinode-y`
   on `main`, README's decomposition paragraph and status row, the
   `npy` remark in `hst_params`, the `nslab == 1` special case in
   `penta_backward` is then the way to get the 1% back.

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
`hst_mpi` (FINDINGS.md).

## Where the time goes, for reference (FINDINGS.md)

One A100, bench_256: cuFFT 31% of the kernel time, `buildrhs` 17%, line
solver 14%, tiled transpose 10%, `build_products` 10%.  Four A100: the
exposed alltoall about 13% of the step.  Two nodes as 4 x 2: the
alltoall phases halve, the y exchange is 45% of the step at 256^3 and
19% at 512^3.  Not to be done on `main`: the `buildrhs`
stencil-and-transpose tile and anything that adds a kernel for a few
percent (the user's call, session 5).

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (also ~/hst, ~/hst-y
and ~/hst-exp on HoreKA as rsync copies of main and of the branch
multinode-y), a GPU/CPU DNS for homogeneous shear turbulence; read
README.md, then NEXT_SESSION.md, then the last two subsections of
FINDINGS.md "The y decomposition".  The y decomposition is measured:
two A100 nodes at 1.28x (256^3) and 1.72x (512^3) one node, with the
y exchange still 45% / 19% of the step.  Two decisions for me to make
first, NEXT_SESSION.md lists them: whether to try the remaining levers
on the exchange (each measured as an A/B in one two-node job), and
whether the branch becomes main.  Branches differ only in hst_mpi and
hst_linsolve; merge main into the branch, never the reverse; docs on
main.  Ask before any change that does not fit that.  Safety net green
after every code change (CPU, GPU, NCCL on istmcetus).  Do not modify
~/Codes/hst/channel or ~/Codes/hst/hst-main.  Commit each step; push
both branches at the end.  First thing: `! ssh horeka true` in the
prompt to open the HoreKA connection.
```
