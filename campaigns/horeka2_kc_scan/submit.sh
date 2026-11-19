#!/bin/bash
# Submit the campaign from a copy of this directory on the HoreKA2 workspace
# (the runs write several GB).  Run it on a login node.
#
#   export HST_ACCOUNT=<your Slurm project account>
#   export HST_PARTITION=<the GPU partition, e.g. an H100/H200 one>
#   ./submit.sh test      4 runs per 4 GPUs (one case), 8 minutes: tests the environment,
#                         the launch and gives s/step.  Work in test/, not in runs/.
#                         Uses HST_TEST_PARTITION if set (a development partition), else HST_PARTITION.
#   ./submit.sh           one job per case in runs/, each with the four angles on four GPUs
#                         (with HST_GPUS=gpu:8 one job with all eight runs)
#
# Optional: HST_ROOT (repository, default ~/hst), TIME (time limit of the real
# jobs, default 06:00:00; unfinished runs continue in a chained job),
# HST_GPUS (GRES string, default gpu:4; the number after the last colon is the
# number of GPUs, and of runs, per job).
#   gpu-h100, gpu-h200, gpu-b200   4 GPUs per node: the default, one job per case
#   gpu-h200-8                     8 GPUs per node: HST_GPUS=gpu:8, all 8 runs in one job
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
# A start file named in a deck must exist here: the code does not stop on a missing
# restart_file, it silently generates the random field of &init instead.
for deck in runs/*/theta_*/hst.in; do
  rf=$(sed -n "s/^ *restart_file *= *'\([^']*\)'.*/\1/p" "$deck")
  { [ -z "$rf" ] || [ "$rf" = Dati.cart.out ]; } && continue
  ( cd "$(dirname "$deck")" && [ -f "$rf" ] ) || { echo "$deck: start file '$rf' not found: fix init_file in make_runs.py, run it, and check the path on this system"; exit 1; }
done
export HST_ROOT
HST_GPUS=${HST_GPUS:-gpu:4}
ngpu=${HST_GPUS##*:}
opts="--gres=$HST_GPUS --ntasks-per-node=$ngpu"

if [ "${1:-}" = test ]; then
  # the first ngpu/4 cases (4 runs each), copied: the test works in test/, not in runs/
  rm -rf test; mkdir -p test
  for c in $(ls runs | head -n $((ngpu / 4))); do cp -r runs/$c test/; done
  runs=$(ls -d test/*/theta_* | tr '\n' ' ')
  # no t_max change needed: wall_max (8 min - 2 min margin) ends the run, MAXSEG=1 prevents the chain
  RUNS="$runs" MARGIN=120 MAXSEG=1 sbatch --export=ALL --job-name=hst-test \
      --account=$HST_ACCOUNT --partition=${HST_TEST_PARTITION:-$HST_PARTITION} $opts --time=00:08:00 horeka2_group.slurm
  echo "after it: grep -h s/step test/*/theta_*/run-*.log | tail -4;  ./status.sh test"
  exit 0
fi

if [ "$ngpu" -ge 8 ]; then          # one job with every run
  runs=$(ls -d runs/*/theta_* | tr '\n' ' ')
  RUNS="$runs" sbatch --export=ALL --job-name=hst-kc \
      --account=$HST_ACCOUNT --partition=$HST_PARTITION $opts --time=$TIME horeka2_group.slurm
else                                  # one job per case
  for c in $(ls runs); do
    runs=$(ls -d runs/$c/theta_* | tr '\n' ' ')
    RUNS="$runs" sbatch --export=ALL --job-name=hst-$c \
        --account=$HST_ACCOUNT --partition=$HST_PARTITION $opts --time=$TIME horeka2_group.slurm
  done
fi
echo "squeue -u \$USER;  ./status.sh"
