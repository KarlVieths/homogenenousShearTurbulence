#!/bin/bash
# Submit the campaign from a copy of this directory on the HoreKA2 workspace
# (the runs write several GB).  Run it on a login node.
#
#   export HST_ACCOUNT=<your Slurm project account>
#   export HST_PARTITION=<the GPU partition, e.g. an H100/H200 one>
#   ./submit.sh test      one case, the four angles, 8 minutes: tests the environment,
#                         the launch and gives s/step.  Work in test/, not in runs/.
#                         Uses HST_TEST_PARTITION if set (a development partition), else HST_PARTITION.
#   ./submit.sh           one job per case in runs/, each with the four angles on four GPUs
#
# Optional: HST_ROOT (repository, default ~/hst), TIME (time limit of the real
# jobs, default 06:00:00; unfinished runs continue in a chained job),
# HST_GPUS (GRES string, default gpu:4).
set -e
cd "$(dirname "$(readlink -f "$0")")"
HST_ROOT=${HST_ROOT:-$HOME/hst}
TIME=${TIME:-06:00:00}
: "${HST_ACCOUNT:?set HST_ACCOUNT (Slurm project account)}"
: "${HST_PARTITION:?set HST_PARTITION (GPU partition)}"
[ -r "$HST_ROOT/env/horeka2.sh" ] || { echo "no $HST_ROOT/env/horeka2.sh: set HST_ROOT to the repository"; exit 1; }
ls -d $HST_ROOT/build-gpu-*/hst >/dev/null 2>&1 || echo "warning: no $HST_ROOT/build-gpu-*/hst yet (run compile-horeka2.sh)"
[ -d runs ] || python3 make_runs.py >/dev/null
mkdir -p logs
export HST_ROOT
gres=--gres=${HST_GPUS:-gpu:4}

if [ "${1:-}" = test ]; then
  case=$(ls runs | head -1)
  rm -rf test; mkdir -p test; cp -r runs/$case test/
  runs=$(ls -d test/$case/theta_* | tr '\n' ' ')
  # no t_max change needed: wall_max (8 min - 2 min margin) ends the run, MAXSEG=1 prevents the chain
  RUNS="$runs" MARGIN=120 MAXSEG=1 sbatch --export=ALL --job-name=hst-test \
      --account=$HST_ACCOUNT --partition=${HST_TEST_PARTITION:-$HST_PARTITION} $gres --time=00:08:00 horeka2_group.slurm
  echo "after it: grep -h s/step test/$case/theta_*/run-*.log | tail -4;  ./status.sh test"
  exit 0
fi

for case in $(ls runs); do
  runs=$(ls -d runs/$case/theta_* | tr '\n' ' ')
  RUNS="$runs" sbatch --export=ALL --job-name=hst-$case \
      --account=$HST_ACCOUNT --partition=$HST_PARTITION $gres --time=$TIME horeka2_group.slurm
done
echo "squeue -u \$USER;  ./status.sh"
