# Next session: read the three two-node jobs, keep the levers with a number

Copy the block at the end as the opening message of the next session.

## Where things stand (2026-09-28, session 9)

Handoff items 1 and 2 are done and committed on both branches:

- `tests/crossbranch.sh <build A> <build B> [nranks] [npy A] [npy B]`
  (`main`): 25 steps with one build, 25 with the other restarted from
  its `Dati.cart.out`, against the 50-step reference.  `main` <-> the
  branch at `npy = 2`, both directions: 2e-14 CPU, 6e-14 GPU.
- `npy = 0` (the default in `hst_input`, `hst.in`, README) means "the
  code chooses": one slab on `main`; on the branch one slab per node
  (ranks per node from `MPI_Comm_split_type`, once, in
  `setup_decomposition`) when the nodes are equal and node-consecutive
  in `MPI_COMM_WORLD`, at most `NPY_MAX = 8` (now in `hst_mpi`), and
  the layout fits the grid; otherwise one slab with a notice.  An
  explicit `npy > 1` with non-consecutive ranks aborts.  Verified on
  istmcetus (2 ranks: 2 x 1) and across istmio2 + istmcetus with the
  system OpenMPI (`mpirun -np 4 -H localhost:2,istmcetus:2 --map-by
  ppr:2:node`, from a directory in the NFS home: 2 x 2; `--map-by
  node`: abort with `npy = 2`, notice and 4 x 1 with `npy = 0`).
  FINDINGS.md "The layout without user input".

Two exchange levers are implemented, validated (safety net green on
CPU at 2 and 4 slabs, GPU-MPI on the RTX 3060, NCCL on istmcetus; the
regression decks also with several solver batches) and **not yet
measured**; each lives in a git worktree on a throwaway branch:

- `~/Codes/hst/homogenenousShearTurbulence-exp`, branch `lever-a`:
  the four ghost-row sends/receives in one NCCL group (two lines).
- `~/Codes/hst/homogenenousShearTurbulence-exp3`, branch `lever-c`:
  `allgather_y` -> `allgather_y_start`/`allgather_y_wait` on the
  communication stream with events (`ev_rec`, `ev_gath`; one
  send/receive per other slab, since a workspace's blocks are not
  contiguous over the slabs; MPI fallback as a shift loop, the first
  version deadlocked at 4 slabs), two solver workspaces when `npy > 1`
  (line offset `w*nlines_max`, the device routines untouched), and
  `line_solve` as a two-deep pipeline over the batches (forward sweep
  of batch b+1 issued before the backward sweep of b).  The batch
  count stays under `line_chunk` (default one batch: halving the batch
  at 256^3 would underfill the A100, 8160 lines = 64 blocks on 108
  SMs), so the overlap needs `line_chunk = nxB/2` in the deck:
  `jobs/horeka_2node.slurm` takes `lc<N>` as the 7th CONFIGS field.
  The `timing` line now reports the *exposed* part of the gather.

Worktrees: `~/Codes/hst/homogenenousShearTurbulence` = `multinode-y`,
`...-main` = `main`, `...-exp` = `lever-a`, `...-exp3` = `lever-c`
(`git worktree list`).  Roots on HoreKA: `~/hst` = `main`, `~/hst-y` =
the branch before `npy = 0` (job 5168845 queued on it: do not rebuild
before it ran), `~/hst-exp` = the branch head, `~/hst-exp2` = lever-a,
`~/hst-exp3` = lever-c *without* the MPI-fallback fix (the job uses
NCCL, unaffected; rsync `...-exp3/` there after job 5168954 ran).

## The three jobs (all `accelerated`, 2 nodes, submitted 2026-09-28
## 11:00-12:00, Slurm's estimate: the night to 2026-09-29)

Output `~/hst-2node-<jobid>.out` in the directory sbatch was run from
(`~/hst-y`, `~/hst-exp`, `~/hst-exp3`); run directories under
`~/hst-runs/2node-*`.

1. **5168845, the fixed cost** (`~/hst-y`, `NPY_REG=2`):
   `8:small:50:nccl:none:2 4:small:50:nccl:none:2 4:small:50:nccl:none:1
   4:bench_256:20:nccl:none:1 8:bench_256:20:nccl:none:2
   4:bench_512:10:nccl:none:1 8:bench_512:10:nccl:none:2`.  Read the
   "y exchange (transfers only)" line of the `small` run on 8 ranks:
   its exchanges carry almost no bytes (2 x 2 KB ghost rows, 28 KB
   records per rank), so ghost rows / 12 and records / 9 are the
   per-exchange latency plus skew between the nodes; the one-node 2 x 2
   row gives the same over NVLink.  If it is close to 0.5 ms, the 21
   exchanges of the bench steps are pure fixed cost and the levers
   below buy only the bandwidth part (512^3).
2. **5168929, lever (a) and `npy = 0` on HoreKA** (`ROOTS="~/hst-exp
   ~/hst-exp2"`, `NPY_REG=0`, all CONFIGS with `npy 0`: `4:bench_256
   8:bench_256 4:bench_512 8:bench_512 8:small:50`).  Check first that
   the banner says `4 x-z pencils x 1 y slabs` on 4 ranks and `4 x 2`
   on 8, and that the regression decks pass at `npy = 0`; then the
   ghost-row line of `~/hst-exp2` against `~/hst-exp` on 8 ranks (5.2
   ms per step at 256^3 before, 8.8 at 512^3).  Keep (a) if the
   two-node step moves beyond the noise (compare only within this job)
   and the 4-rank rows stay a wash: `cd ...-exp && git commit -am ...`,
   then `cd homogenenousShearTurbulence && git merge lever-a`; else
   `git worktree remove ...-exp && git branch -D lever-a`.
3. **5168954, lever (c)** (`ROOTS="~/hst-exp ~/hst-exp3"`, `NPY_REG=0`):
   `4:bench_256 8:bench_256 8:bench_256:lc16 4:bench_512 8:bench_512
   8:bench_512:lc32 8:bench_512:lc16`.  The baseline root's `lc` rows
   show the cost of splitting the batch alone, lever-c's `lc` rows the
   overlap (records 14.9 ms of the 126.5 ms step at 512^3, of which
   about 10 is bandwidth; the `lc0` rows of both roots must agree).
   Keep it as above if `lc32` at 512^3 beats the baseline's `lc0` by
   more than the noise; then decide the default: either `line_chunk`
   stays a deck parameter (document the value in README) or
   `init_linsolve` takes two batches when `npy > 1` and the batch keeps
   at least ~14000 lines (108 SMs x 128 threads) - the user prefers
   the least input, but no autotuner.

After the merges: `git merge main` into the branch (docs), the safety
net on both (below), FINDINGS.md (a subsection per lever with the
table, after "The layout without user input"), README's performance
table and its 2 x 4 A100 column, this file; push both branches; rsync
the branch to `~/hst-y` and `~/hst-exp` and `main` to `~/hst`.

## Levers left

- (b) the two ghost exchanges that follow each other (`fill_ghosts(1)`
  then `fill_ghosts(3)` in `hst_equations`) as one exchange of two
  fields: 12 to 9 exchanges a step, i.e. 3 x the fixed cost.  Needs a
  `fill_ghosts(c1, c2)` in `hst_derivatives` and a change in
  `hst_equations`, so it touches physics files on `main`: **ask the
  user first** (not done in session 9 for that reason).
- (d) smaller records for the kinds whose matrix does not change
  between calls: bytes only, 512^3 only; after (c) has shown how much
  of the record time is exposed.

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

The CPU build must be made in a shell that has *not* sourced
`env/istm.sh`, the GPU tests in one that *has*; a flag change needs the
objects removed; `build-nccl` is exercised on istmcetus only, where `2 2`
(one pencil x two slabs) uses the NCCL y column and `2` the NCCL
alltoall.  Test decks with more than one solver batch: `line_chunk = 3`
or `5` in `&mesh` (nxB = 8 or 16).  A bit-for-bit CPU comparison needs
`FFTW_ESTIMATE`.  nvfortran 25.9 rejects the names `kind` and `x` in
OpenMP clauses of any file that can see `hst_mpi`.  Not on `main`: the
`buildrhs` tile and anything that adds a kernel for a few percent.
Scripted ssh to HoreKA: `ssh -o BatchMode=yes -o ProxyCommand=false
horeka ...` (a refused mux session then fails without a login attempt;
HoreKA counts failed logins); `ssh -O check horeka` says whether the
master is alive; when it is not, the user types `! ssh horeka true`.

## Opening message for the next session

```
Repository ~/Codes/hst/homogenenousShearTurbulence (branch multinode-y;
~/Codes/hst/homogenenousShearTurbulence-main is a worktree of main,
...-exp and ...-exp3 hold the unmeasured levers lever-a and lever-c),
a GPU/CPU DNS for homogeneous shear turbulence; read README.md, then
NEXT_SESSION.md, then FINDINGS.md "The layout without user input".
Task: read the three HoreKA two-node jobs listed in NEXT_SESSION.md
(5168845 fixed cost, 5168929 lever (a) + npy = 0 on two nodes, 5168954
lever (c)); keep each lever only with the number (two-node step beyond
the noise, one-node rows a wash), merge what is kept into multinode-y,
docs on main (FINDINGS subsection per lever with its table, README
performance table, NEXT_SESSION.md), merge main into the branch, never
the reverse; safety net green after every code change (CPU, GPU, NCCL
on istmcetus, the cross-branch round trip); rsync the kept state to
HoreKA (~/hst = main, ~/hst-y = ~/hst-exp = branch).  Then lever (d) if
(c) says the records' bandwidth is exposed; lever (b) touches physics
files, ask me first.  Do not modify ~/Codes/hst/channel or
~/Codes/hst/hst-main.  Commit each step; push both branches at the end.
First thing: `! ssh horeka true` in the prompt.
```
