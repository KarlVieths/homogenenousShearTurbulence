#!/bin/bash
# HoreKA: up to four campaign runs on one A100 node, one per GPU, concurrently.
#
#   cd $CAMP && sbatch --export=ALL,RUNS="runs/hex3/theta_15.0 runs/rect/theta_15.0 ..." --time=02:00:00 tools/horeka_group.slurm
#
# CAMP = the campaign directory on HoreKA (this script's ../), EXE = its
# bin/hst-a100.  Every run gets wall_max = job limit - MARGIN (the restart
# file is written before the limit); a run that already has Runtimedata and
# Dati.cart.out is resumed (time_from_restart = .true.).  At the end each run
# is marked DONE (t_max reached), INCOMPLETE (stopped by wall_max: submit it
# again) or FAILED.  mpirun --bind-to none: HPC-X binds a one-rank job to
# core 0, and the four runs would share it.
#SBATCH --job-name=dp-group
#SBATCH --account=hk-project-exasim
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/dp-group-%j.out

CAMP=$(cd "$SLURM_SUBMIT_DIR" && pwd)
EXE=${EXE:-$CAMP/bin/hst-a100}
MARGIN=${MARGIN:-300}
[ -z "$RUNS" ] && { echo "RUNS must be set"; exit 1; }
source $HOME/hst/env/horeka.sh gpu > /dev/null 2>&1

limit=$(squeue -h -j $SLURM_JOB_ID -o %l)
days=0; case "$limit" in *-*) days=${limit%%-*}; limit=${limit#*-};; esac
IFS=: read -r h m s <<< "$limit"; [ -z "$s" ] && { s=$m; m=$h; h=0; }
wall=$((days*86400 + 10#$h*3600 + 10#$m*60 + 10#$s - MARGIN))
echo "job $SLURM_JOB_ID on $(hostname), limit $(squeue -h -j $SLURM_JOB_ID -o %l), wall_max = $wall s, exe $EXE"

i=0
for r in $RUNS; do
  d=$CAMP/$r
  sed -i "s/wall_max = [0-9.eE+-]*/wall_max = $wall/" $d/hst.in
  if [ -f $d/Runtimedata ] && [ -f $d/Dati.cart.out ]; then
    sed -i 's/time_from_restart = \.false\./time_from_restart = .true./' $d/hst.in
    [ -f $d/run.log ] && mv $d/run.log $d/run-$(date +%s).log
    echo "GPU $i: resume $r"
  else
    echo "GPU $i: start $r"
  fi
  rm -f $d/INCOMPLETE $d/FAILED
  ( cd $d && CUDA_VISIBLE_DEVICES=$i mpirun --bind-to none -np 1 $EXE hst.in > run.log 2>&1 ) &
  i=$((i + 1))
done
wait

for r in $RUNS; do
  d=$CAMP/$r
  tmax=$(sed -n 's/^ *t_max *= *\([0-9.eE+-]*\).*/\1/p' $d/hst.in)
  tend=$(sed -n 's/.*end of run at time *\([0-9.eE+-]*\).*/\1/p' $d/run.log)
  sps=$(grep -o '[0-9.]* s/step' $d/run.log | tail -1)
  if [ -n "$tend" ] && awk -v a="$tend" -v b="$tmax" 'BEGIN{exit !(a >= b - 0.01)}'; then
    touch $d/DONE; echo "done       $r (t = $tend, $sps)"
  elif grep -q "wall-clock limit" $d/run.log; then
    touch $d/INCOMPLETE; echo "INCOMPLETE $r (t = $tend, $sps): submit again"
  else
    touch $d/FAILED; echo "FAILED     $r (end time '$tend')"; tail -5 $d/run.log
  fi
done
