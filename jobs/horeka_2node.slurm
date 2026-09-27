#!/bin/bash
# HoreKA two-node measurement: the same benchmarks on 4 A100 of one node and
# on 8 A100 of two nodes (InfiniBand between them), with the per-phase timer.
# Configurations: transport nccl / nccl2 (two-level) / mpi, ranks bound to
# the NUMA domain of their GPU (jobs/bind_numa.sh) or not.  The first 8-GPU
# run prints NCCL's network setup (NCCL_DEBUG=INFO: adapter and GDRDMA per
# connection), and the regression decks run on 8 ranks with nccl2 against
# tests/reference.
#   cd ~/hst && sbatch --partition=accelerated jobs/horeka_2node.slurm     (HST_ROOT, RUNS overridable;
#                                                                            dev_accelerated has 3 nodes and rarely two free)
#   sbatch ... --export=ALL,CONFIGS="8:bench_256:20:nccl:none:info",NCCL_NET_GDR_LEVEL=SYS jobs/horeka_2node.slurm
# CONFIGS lists the runs as ranks:deck:steps:transport:bind[:info]; NCCL_*
# variables in the environment reach the runs (e.g. NCCL_NET_GDR_LEVEL=SYS
# lets NCCL use GPUDirect RDMA for GPUs that are not next to the adapter).
#SBATCH --job-name=hst-2node
#SBATCH --account=hk-project-exasim
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=hst-2node-%j.out

HST_ROOT=${HST_ROOT:-$HOME/hst}
RUNS=${RUNS:-$HOME/hst-runs}
source $HST_ROOT/env/horeka.sh gpu
export NCCL_DEBUG=WARN
CONFIGS=${CONFIGS:-"4:bench_256:20:nccl:none 8:bench_256:20:nccl:none:info 8:bench_256:20:nccl:numa 8:bench_256:20:nccl2:none 8:bench_256:20:nccl2:numa 8:bench_256:20:mpi:none 4:bench_512:10:nccl:none 8:bench_512:10:nccl:numa 8:bench_512:10:nccl2:numa"}
env | grep '^NCCL_' | grep -v NCCL_DEBUG
scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd' '
echo "=== topology of $(hostname)"; nvidia-smi topo -m | head -12; numactl -H | grep -E "available|cpus" | cut -c1-80
run() {  # run <ranks> <deck> <nstep> <transport> <bind: none|numa> [info]
  d=$RUNS/2node-$2-np$1; mkdir -p $d; cd $d; rm -f Dati.cart.out Runtimedata
  sed -e "s/nstep = [0-9]*/nstep = $3/" -e "s/time_from_restart = .false./time_from_restart = .false., timing = .true./" \
      -e "s/&mesh /\&mesh transport = '$4', /" $HST_ROOT/examples/$2.in > hst.in
  wrap=; [ "$5" = numa ] && wrap=$HST_ROOT/jobs/bind_numa.sh
  echo "=== $2 on $1 GPU(s), $3 steps, transport $4, bind $5 $6"
  if [ "$6" = info ]; then
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET mpirun -np $1 --map-by ppr:4:node --bind-to none $wrap $HST_ROOT/build-gpu/hst hst.in > run.log 2>&1
    grep -E "NCCL version|NET/IB : Using|Using network|via NET|GDRDMA|NCCL INFO Channel 00/0" run.log | sed 's/.*NCCL INFO //' | sort | uniq -c | sort -rn | head -12
    grep -E -A11 "s/step; writing|rror|time per phase|transport|bind_numa" run.log
  else
    mpirun -np $1 --map-by ppr:4:node --bind-to none $wrap $HST_ROOT/build-gpu/hst hst.in | grep -E -A11 "s/step; writing|rror|time per phase|transport|bind_numa"
  fi
}
for c in $CONFIGS; do run ${c//:/ }; done
echo "=== regression decks on 8 ranks, transport nccl2"
for deck in small small_s2x small_stokes; do
  d=$RUNS/2node-reg-$deck; mkdir -p $d; cd $d; rm -f Dati.cart.out
  sed -e "s/&mesh /\&mesh transport = 'nccl2', /" $HST_ROOT/tests/decks/$deck.in > hst.in
  mpirun -np 8 --map-by ppr:4:node --bind-to none $HST_ROOT/build-gpu/hst hst.in > run.log 2>&1 || echo "$deck: run failed"
  echo "--- $deck"; python3 $HST_ROOT/tests/compare_fields.py Dati.cart.out $HST_ROOT/tests/reference/$deck.fld 1e-10 | tail -1
done
