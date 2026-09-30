#!/bin/bash
# HoreKA production run in segments: one MPI rank per GPU on the job's nodes
# (npy = 0 in the deck: one y slab per node), the deck copied to RUNDIR/hst.in
# with wall_max = the job's time limit minus MARGIN seconds (the final restart
# file) and time_from_restart = .true., and the job resubmitted from
# Dati.cart.out until the run reaches t_max (up to MAXSEG segments).
#
#   sbatch --export=ALL,DECK=$HOME/hst/examples/prod_re20000.in,RUNDIR=/hkfs/work/workspace/scratch/xt8786-hst/re20000 \
#          --nodes=2 --time=12:00:00 ~/hst/jobs/horeka_prod.slurm
#   sbatch --export=ALL,DECK=$HOME/hst/hst.in,RUNDIR=$HOME/hst-runs/prod-small \
#          --partition=dev_accelerated --nodes=1 --time=00:50:00 ~/hst/jobs/horeka_prod.slurm      (the small deck, one node)
#
# The time limit is the expected length of the whole run plus a margin when
# one segment can hold it, else the longest the partition allows; every
# segment writes run-<jobid>.log in RUNDIR (steps, snapshot and restart-file
# times) and the segment's summary goes to hst-prod-<jobid>.out here.
# HST_ROOT (~/hst), BUILD (build-gpu), MARGIN (600 s) and MAXSEG (30) are
# overridable.  To stop the chain: scancel the running or pending job.
#SBATCH --job-name=hst-prod
#SBATCH --account=hk-project-exasim
#SBATCH --partition=accelerated
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=hst-prod-%j.out

HST_ROOT=${HST_ROOT:-$HOME/hst}
BUILD=${BUILD:-build-gpu}
MARGIN=${MARGIN:-600}
MAXSEG=${MAXSEG:-30}
SEG=${SEG:-1}
if [ -z "$DECK" ] || [ -z "$RUNDIR" ]; then echo "DECK and RUNDIR must be set (--export=ALL,DECK=...,RUNDIR=...)"; exit 1; fi
source $HST_ROOT/env/horeka.sh gpu
export NCCL_DEBUG=WARN

# the time limit of this job in seconds ([D-]HH:MM:SS from squeue)
limit=$(squeue -h -j $SLURM_JOB_ID -o %l)
days=0; case "$limit" in *-*) days=${limit%%-*}; limit=${limit#*-};; esac
IFS=: read -r h m s <<< "$limit"; [ -z "$s" ] && { s=$m; m=$h; h=0; }
wall=$((days*86400 + 10#$h*3600 + 10#$m*60 + 10#$s - MARGIN))

mkdir -p $RUNDIR; cd $RUNDIR
sed -e "s/wall_max = [0-9.eE+-]*/wall_max = $wall/" -e "s/time_from_restart = \.false\./time_from_restart = .true./" $DECK > hst.in
grep -q "wall_max = $wall" hst.in || { echo "the deck must contain wall_max = <value> in &time_control"; exit 1; }
echo "segment $SEG of job $SLURM_JOB_ID on $(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd' '), $SLURM_NTASKS ranks"
echo "deck $DECK -> $RUNDIR/hst.in, wall_max = $wall s (limit $(squeue -h -j $SLURM_JOB_ID -o %l) minus $MARGIN s)"
[ -f Dati.cart.out ] && echo "restarting from Dati.cart.out ($(stat -c %s Dati.cart.out) bytes, $(stat -c %y Dati.cart.out))"

log=run-$SLURM_JOB_ID.log
mpirun -np $SLURM_NTASKS --map-by ppr:$SLURM_NTASKS_PER_NODE:node --bind-to none $HST_ROOT/$BUILD/hst hst.in > $log 2>&1
grep -E "ranks =|transport|restart file|generating|snapshots written|restart file written|end of run|wall-clock|rror|failed" $log | tail -40
grep -E "s/step" $log | tail -3

# next segment
if ! grep -q "end of run at time" $log; then echo "the run did not end normally: no resubmission"; exit 1; fi
if ! grep -q "wall-clock limit" $log; then echo "the run reached t_max: done"; exit 0; fi
if [ $SEG -ge $MAXSEG ]; then echo "MAXSEG = $MAXSEG segments reached: no resubmission"; exit 1; fi
cd $SLURM_SUBMIT_DIR
next=$(sbatch --parsable --partition=$SLURM_JOB_PARTITION --nodes=$SLURM_JOB_NUM_NODES \
       --time=$(squeue -h -j $SLURM_JOB_ID -o %l) --export=ALL,DECK=$DECK,RUNDIR=$RUNDIR,SEG=$((SEG+1)) $HST_ROOT/jobs/horeka_prod.slurm)
echo "segment $((SEG+1)) submitted as job $next"
