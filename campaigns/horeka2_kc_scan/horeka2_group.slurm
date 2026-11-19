#!/bin/bash
# HoreKA2: up to four independent single-GPU runs on one node, one per GPU, at
# the same time.  Job script of the campaign; submit it with submit.sh, which
# supplies the account, partition and run list.
#
# What it does, per run directory in $RUNS (paths relative to the submit directory):
#   * wall_max in the deck = job time limit - MARGIN, so that the run stops
#     and writes its restart file Dati.cart.out before Slurm kills the job;
#   * if Dati.cart.out exists, time_from_restart = .true. (the run resumes);
#   * the log of this job's run is run-<jobid>.log in the run directory;
#   * at the end every run is marked DONE (t_max reached), INCOMPLETE (stopped
#     by wall_max) or FAILED (anything else) by an empty file of that name;
#   * the INCOMPLETE runs are resubmitted by this script, up to MAXSEG jobs
#     in a chain.  scancel the pending job to stop the chain.
#
# Variables (environment, normally set by submit.sh):
#   RUNS      required, space-separated run directories, at most one per GPU
#   HST_ROOT  repository with env/horeka2.sh and the builds (default $HOME/hst)
#   EXE       the binary; default $HST_ROOT/build-gpu-<device>/hst with the
#             device (h100, h200, b200) from the GPU's name; an H100 or H200
#             node without its own build directory uses build-gpu-h100
#   MARGIN    seconds before the limit at which a run stops (default 300)
#   MAXSEG    maximum number of chained jobs (default 10), SEG the current one
#   LAUNCH    how a run is started (default "mpirun --bind-to none -np 1";
#             LAUNCH= starts the binary directly, MPI then runs as one process)
#
#SBATCH --job-name=hst-kc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

CAMP=$SLURM_SUBMIT_DIR
HST_ROOT=${HST_ROOT:-$HOME/hst}
MARGIN=${MARGIN:-300}
MAXSEG=${MAXSEG:-10}
SEG=${SEG:-1}
LAUNCH=${LAUNCH-mpirun --bind-to none -np 1}
[ -z "$RUNS" ] && { echo "RUNS must be set (use submit.sh)"; exit 1; }

# The environment.  On the GPU node the architecture is detected with nvidia-smi.
source $HST_ROOT/env/horeka2.sh gpu || { echo "env/horeka2.sh failed"; exit 1; }
if [ -z "$EXE" ]; then
  # build-gpu-<device> by the GPU's name; cc90 without a build of its own uses build-gpu-h100
  gname=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr 'A-Z' 'a-z')
  case "$gname" in *h200*) dev=h200;; *h100*) dev=h100;; *b200*) dev=b200;; *) dev=;; esac
  [ -z "$dev" ] && case "$GPU_ARCH" in cc90) dev=h100;; cc100) dev=b200;; esac
  [ -x $HST_ROOT/build-gpu-$dev/hst ] || { [ "$GPU_ARCH" = cc90 ] && dev=h100; }
  [ -n "$dev" ] || { echo "GPU '$gname' ($GPU_ARCH): no binary known, set EXE"; exit 1; }
  EXE=$HST_ROOT/build-gpu-$dev/hst
fi
[ -x "$EXE" ] || { echo "binary $EXE not found (build it with compile-horeka2.sh)"; exit 1; }

# Pre-flight: does CUDA initialise on this node?  (nvidia-smi can list GPUs that CUDA cannot use.)
nva=$NVHPC_ROOT/compilers/bin/nvaccelinfo
if [ -x $nva ] && ! $nva 2>&1 | grep -q "Device Number"; then
  echo "CUDA does not initialise on $(hostname): nvaccelinfo sees no accelerator."
  $nva 2>&1 | tail -4
  echo "Report the node to the cluster support; resubmit with SBATCH_EXCLUDE=$(hostname) in the environment."
  exit 1
fi

ngpu=$(nvidia-smi -L | wc -l)
nrun=$(echo $RUNS | wc -w)
[ "$nrun" -le "$ngpu" ] || { echo "$nrun runs but only $ngpu GPUs on this node"; exit 1; }

# the time limit of this job in seconds ([D-]HH:MM:SS from squeue)
limit=$(squeue -h -j $SLURM_JOB_ID -o %l)
days=0; case "$limit" in *-*) days=${limit%%-*}; limit=${limit#*-};; esac
IFS=: read -r h m s <<< "$limit"; [ -z "$s" ] && { s=$m; m=$h; h=0; }
wall=$((days*86400 + 10#$h*3600 + 10#$m*60 + 10#$s - MARGIN))
echo "job $SLURM_JOB_ID (segment $SEG) on $(hostname), $ngpu GPUs ($GPU_ARCH), limit $(squeue -h -j $SLURM_JOB_ID -o %l), wall_max = $wall s"
echo "binary $EXE"
nvidia-smi -L

log=run-$SLURM_JOB_ID.log
i=0
started=
for r in $RUNS; do
  d=$CAMP/$r
  [ -f $d/hst.in ] || { echo "no deck $d/hst.in"; exit 1; }
  if [ -f $d/DONE ]; then echo "skip   $r (DONE)"; continue; fi
  sed -i "s/wall_max = [0-9.eE+-]*/wall_max = $wall/" $d/hst.in
  grep -q "wall_max = $wall" $d/hst.in || { echo "$d/hst.in has no 'wall_max = <value>'"; exit 1; }
  if [ -f $d/Dati.cart.out ]; then
    sed -i -e 's/time_from_restart = \.false\./time_from_restart = .true./' \
           -e "s/restart_file = '[^']*'/restart_file = 'Dati.cart.out'/" $d/hst.in
    echo "GPU $i: resume $r ($(stat -c %y $d/Dati.cart.out | cut -c1-19))"
  else
    echo "GPU $i: start  $r"
  fi
  rm -f $d/INCOMPLETE $d/FAILED
  ( cd $d && CUDA_VISIBLE_DEVICES=$i $LAUNCH $EXE hst.in > $log 2>&1 ) &
  started="$started $r"
  i=$((i + 1))
done
wait

# classify the runs of this job
incomplete=
for r in $started; do
  d=$CAMP/$r
  tmax=$(sed -n 's/^ *t_max *= *\([0-9.eE+-]*\).*/\1/p' $d/hst.in)
  tend=$(sed -n 's/.*end of run at time *\([0-9.eE+-]*\) after.*/\1/p' $d/$log)
  sps=$(grep -o '[0-9.]* s/step' $d/$log | tail -1)
  if [ -n "$tend" ] && awk -v a="$tend" -v b="$tmax" 'BEGIN{exit !(a >= b - 0.01)}'; then
    touch $d/DONE;       echo "done       $r (t = $tend, $sps)"
  elif grep -q "wall-clock limit" $d/$log; then
    touch $d/INCOMPLETE; echo "INCOMPLETE $r (t = $tend of $tmax, $sps)"; incomplete="$incomplete $r"
  else
    touch $d/FAILED;     echo "FAILED     $r (end time '$tend')"; tail -5 $d/$log
  fi
done

# chain: the runs that the wall-clock limit stopped
if [ -n "$incomplete" ]; then
  if [ $SEG -ge $MAXSEG ]; then echo "MAXSEG = $MAXSEG segments reached: no resubmission"; exit 1; fi
  cd $CAMP
  next=$(RUNS="${incomplete# }" SEG=$((SEG+1)) sbatch --parsable --export=ALL \
         --job-name=$SLURM_JOB_NAME --account=$SLURM_JOB_ACCOUNT --partition=$SLURM_JOB_PARTITION \
         --time=$(squeue -h -j $SLURM_JOB_ID -o %l) $CAMP/horeka2_group.slurm)
  echo "segment $((SEG+1)) submitted as job $next for:$incomplete"
fi
