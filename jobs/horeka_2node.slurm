#!/bin/bash
# HoreKA two-node measurement: the same benchmarks on 4 A100 of one node and
# on 8 A100 of two nodes (InfiniBand between them), with the per-phase timer,
# NCCL transport (and the MPI transport once, for reference).
#   cd ~/hst && sbatch jobs/horeka_2node.slurm            (HST_ROOT, RUNS overridable)
#SBATCH --job-name=hst-2node
#SBATCH --account=hk-project-exasim
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=hst-2node-%j.out

HST_ROOT=${HST_ROOT:-$HOME/hst}
RUNS=${RUNS:-$HOME/hst-runs}
source $HST_ROOT/env/horeka.sh gpu
export NCCL_DEBUG=WARN
scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd' '
run() {  # run <ranks> <deck> <nstep> <transport>
  d=$RUNS/2node-$2-np$1; mkdir -p $d; cd $d; rm -f Dati.cart.out Runtimedata
  sed -e "s/nstep = [0-9]*/nstep = $3/" -e "s/time_from_restart = .false./time_from_restart = .false., timing = .true./" \
      -e "s/&mesh /\&mesh transport = '$4', /" $HST_ROOT/examples/$2.in > hst.in
  echo "=== $2 on $1 GPU(s), $3 steps, transport $4"
  mpirun -np $1 --map-by ppr:4:node --bind-to none $HST_ROOT/build-gpu/hst hst.in | grep -E -A11 "s/step; writing|rror|time per phase|transport"
}
run 4 bench_256 20 nccl
run 8 bench_256 20 nccl
run 8 bench_256 20 mpi
run 4 bench_512 10 nccl
run 8 bench_512 10 nccl
run 8 bench_512 10 mpi
