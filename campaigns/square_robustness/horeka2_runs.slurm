#!/bin/bash
# HoreKA2: independent runs on one node, each on RUN_GPUS GPUs (default 1: up to four single-GPU
# runs at the same time, one per GPU; RUN_GPUS=4: one run on all four GPUs, MPI ranks = GPUs).
# Job script of the robustness campaign (the script of campaigns/horeka2_kc_scan with runs of several
# GPUs and a STOP file); submit it with submit.py, which supplies the account, partition and run list.
#
# What it does, per run directory in $RUNS (paths relative to the submit directory):
#   * wall_max in the deck = job time limit - MARGIN, so that the run stops
#     and writes its restart file Dati.cart.out before Slurm kills the job;
#   * if Dati.cart.out exists, time_from_restart = .true. (the run resumes);
#   * the log of this job's run is run-<jobid>.log in the run directory;
#   * at the end every run is marked DONE (t_max reached), INCOMPLETE (stopped
#     by wall_max), STOPPED (see STOP) or FAILED (anything else) by an empty file of that name;
#   * the INCOMPLETE runs are resubmitted by this script, up to MAXSEG jobs
#     in a chain.  scancel the pending job to stop the chain.
#   * STOP: `touch runs/<id>/STOP` ends that run within a minute (SIGTERM to its mpirun, no restart
#     file of its own: the last Dati.cart.out, up to dt_save = 100 old, stays) and marks it STOPPED;
#     the other runs of the job go on.  For a run that has gone laminar (stage 3a).  To resume it
#     later remove STOP and STOPPED.
#
# Variables (environment, normally set by submit.py):
#   RUNS      required, space-separated run directories; at most (GPUs on the node)/RUN_GPUS of them
#   RUN_GPUS  GPUs per run (default 1); run k uses GPUs k*RUN_GPUS ... (k+1)*RUN_GPUS - 1 of the node
#   HST_ROOT  repository with env/horeka2.sh and the builds (default $HOME/hst)
#   EXE       the binary; default $HST_ROOT/build-gpu-<device>/hst with the
#             device (h100, h200, b200) from the GPU's name; an H100 or H200
#             node without its own build directory uses build-gpu-h100
#   MARGIN    seconds before the limit at which a run stops (default 300)
#   MAXSEG    maximum number of chained jobs (default 10), SEG the current one
#   STOP_POLL seconds between looks for STOP files (default 60)
#   LAUNCH    how a run is started (default "mpirun --bind-to none -np $RUN_GPUS")
#
#SBATCH --job-name=hst-rob
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
RUN_GPUS=${RUN_GPUS:-1}
LAUNCH=${LAUNCH-mpirun --bind-to none -np $RUN_GPUS}
[ -z "$RUNS" ] && { echo "RUNS must be set (use submit.py)"; exit 1; }

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
# the binary must be built for this node's CPU architecture: the B200 nodes are ARM (Grace), the H100 and H200 nodes x86-64
if command -v file >/dev/null 2>&1; then
  ftype=$(file -L -b "$EXE")
  case "$ftype" in
    *ELF*) case "$(uname -m):$ftype" in
             x86_64:*x86-64*|aarch64:*aarch64*) ;;
             *) echo "binary $EXE is not for this node ($(uname -m)): ${ftype:0:90}"
                echo "build it on this architecture: ./compile-horeka2.sh <device> in the repository builds it on a node of that kind"; exit 1;;
           esac;;
  esac
fi

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
[ $((nrun*RUN_GPUS)) -le "$ngpu" ] || { echo "$nrun runs of $RUN_GPUS GPUs but only $ngpu GPUs on this node"; exit 1; }

# the time limit of this job in seconds ([D-]HH:MM:SS from squeue)
limit=$(squeue -h -j $SLURM_JOB_ID -o %l)
days=0; case "$limit" in *-*) days=${limit%%-*}; limit=${limit#*-};; esac
IFS=: read -r h m s <<< "$limit"; [ -z "$s" ] && { s=$m; m=$h; h=0; }
wall=$((days*86400 + 10#$h*3600 + 10#$m*60 + 10#$s - MARGIN))
echo "job $SLURM_JOB_ID (segment $SEG) on $(hostname), $ngpu GPUs ($GPU_ARCH), $RUN_GPUS per run, limit $(squeue -h -j $SLURM_JOB_ID -o %l), wall_max = $wall s"
echo "binary $EXE"
[ -f "$(dirname $EXE)/BUILD_INFO" ] && sed -n 's/^commit: *//p;s/^built: *//p' "$(dirname $EXE)/BUILD_INFO" | sed 's/^/  build: /'
nvidia-smi -L

log=run-$SLURM_JOB_ID.log
# the GPUs of this job: Slurm's list (renumbered 0,1,.. under cgroups, real indices otherwise), else 0..ngpu-1
vis=(${CUDA_VISIBLE_DEVICES//,/ })
[ ${#vis[@]} -ge $ngpu ] || vis=($(seq 0 $((ngpu - 1))))
i=0
started=
pids=()
for r in $RUNS; do
  d=$CAMP/$r
  [ -f $d/hst.in ] || { echo "no deck $d/hst.in"; exit 1; }
  if [ -f $d/DONE ]; then echo "skip   $r (DONE)"; continue; fi
  if [ -f $d/STOPPED ]; then echo "skip   $r (STOPPED; remove STOP and STOPPED to resume)"; continue; fi
  sed -i "s/wall_max = [0-9.eE+-]*/wall_max = $wall/" $d/hst.in
  grep -q "wall_max = $wall" $d/hst.in || { echo "$d/hst.in has no 'wall_max = <value>'"; exit 1; }
  gl=$(IFS=,; echo "${vis[*]:$i:$RUN_GPUS}")
  if [ -f $d/Dati.cart.out ]; then
    sed -i -e 's/time_from_restart = \.false\./time_from_restart = .true./' \
           -e "s/restart_file = '[^']*'/restart_file = 'Dati.cart.out'/" $d/hst.in
    echo "GPU $gl: resume $r ($(stat -c %y $d/Dati.cart.out | cut -c1-19))"
  else
    echo "GPU $gl: start  $r"
  fi
  rm -f $d/INCOMPLETE $d/FAILED
  [ -f "$(dirname $EXE)/BUILD_INFO" ] && cp "$(dirname $EXE)/BUILD_INFO" $d/BUILD_INFO     # the commit and node of the binary of this run
  # exec: the subshell becomes the launcher, so its pid is the one the STOP watcher signals
  ( cd $d && CUDA_VISIBLE_DEVICES=$gl exec $LAUNCH $EXE hst.in > $log 2>&1 ) &
  pids+=($!)
  started="$started $r"
  i=$((i + RUN_GPUS))
done

# the STOP files: end that run, the others go on
watch() {
  local k
  while sleep ${STOP_POLL:-60}; do
    k=0
    for r in $started; do
      if [ -f $CAMP/$r/STOP ] && kill -0 ${pids[$k]} 2>/dev/null; then
        echo "$(date +%T) STOP file of $r: terminating it"; kill -TERM ${pids[$k]}
      fi
      k=$((k + 1))
    done
  done
}
watch &
wpid=$!
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
kill $wpid 2>/dev/null

# classify the runs of this job
incomplete=
for r in $started; do
  d=$CAMP/$r
  tmax=$(sed -n 's/^ *t_max *= *\([0-9.eE+-]*\).*/\1/p' $d/hst.in)
  tend=$(sed -n 's/.*end of run at time *\([0-9.eE+-]*\) after.*/\1/p' $d/$log)
  sps=$(grep -o '[0-9.]* s/step' $d/$log | tail -1)
  if [ -f $d/STOP ] && [ -z "$tend" ]; then
    touch $d/STOPPED;    echo "STOPPED    $r (by its STOP file, last restart file kept)"
  elif [ -n "$tend" ] && awk -v a="$tend" -v b="$tmax" 'BEGIN{exit !(a >= b - 0.01)}'; then
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
  gres=$(squeue -h -j $SLURM_JOB_ID -o %b | sed 's#^gres[/:]##'); [ -n "$gres" ] || gres=gpu:$ngpu
  next=$(RUNS="${incomplete# }" SEG=$((SEG+1)) sbatch --parsable --export=ALL \
         --job-name=$SLURM_JOB_NAME --account=$SLURM_JOB_ACCOUNT --partition=$SLURM_JOB_PARTITION \
         --gres=$gres --ntasks-per-node=${SLURM_NTASKS_PER_NODE:-$ngpu} \
         --time=$(squeue -h -j $SLURM_JOB_ID -o %l) $CAMP/horeka2_runs.slurm)
  echo "segment $((SEG+1)) submitted as job $next for:$incomplete"
fi
