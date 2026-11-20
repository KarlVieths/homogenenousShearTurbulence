#!/bin/bash
# Submit the seed ensemble from a copy of this directory on the HoreKA2 workspace
# (the runs write about 10-13 GB each).  Run it on a login node.
#
#   export HST_ACCOUNT=<your Slurm project account>
#   export HST_PARTITION=<the GPU partition, 4 GPUs per node: gpu-h100, gpu-h200 or gpu-b200>
#   ./submit.sh test      the four runs for 8 minutes: starts, CFL step, s/step, seeds, first q2.
#                         Works in test/, not in runs/.  HST_TEST_PARTITION (a development
#                         partition, e.g. dev-gpu-h100) replaces HST_PARTITION for it.
#   ./submit.sh           one job with the four runs, one per GPU of a node
#
# Optional: HST_ROOT (repository, default ~/hst), TIME (time limit, default 12:00:00;
# an unfinished run continues in a chained job), HST_GPUS (GRES string, default gpu:4).
set -e
cd "$(dirname "$(readlink -f "$0")")"
HST_ROOT=${HST_ROOT:-$HOME/hst}
TIME=${TIME:-12:00:00}
: "${HST_ACCOUNT:?set HST_ACCOUNT (Slurm project account)}"
: "${HST_PARTITION:?set HST_PARTITION (GPU partition)}"
[ -r "$HST_ROOT/env/horeka2.sh" ] || { echo "no $HST_ROOT/env/horeka2.sh: set HST_ROOT to the repository"; exit 1; }
ls -d $HST_ROOT/build-gpu-*/hst >/dev/null 2>&1 || echo "warning: no $HST_ROOT/build-gpu-*/hst yet (run compile-horeka2.sh)"
[ -d runs ] || python3 make_runs.py >/dev/null
mkdir -p logs
export HST_ROOT
HST_GPUS=${HST_GPUS:-gpu:4}
ngpu=${HST_GPUS##*:}
opts="--gres=$HST_GPUS --ntasks-per-node=$ngpu"

# the four decks must differ in the seed (and the header line) only
first=$(ls -d runs/*/ | head -n1)
for d in runs/*/; do
  n=$(diff <(tail -n +2 $first/hst.in) <(tail -n +2 $d/hst.in) | grep -c '^[<>]' || true)
  [ "$d" = "$first" ] || [ "$n" -eq 2 ] || { echo "$d/hst.in differs from $first/hst.in in more than the seed line: run ./check.sh decks"; exit 1; }
done
nrun=$(ls -d runs/*/ | wc -l)
[ "$nrun" -le "$ngpu" ] || { echo "$nrun runs but $ngpu GPUs per job (HST_GPUS)"; exit 1; }

if [ "${1:-}" = test ]; then
  rm -rf test; mkdir -p test
  cp -r runs/* test/
  runs=$(ls -d test/*/ | sed 's#/$##' | tr '\n' ' ')
  # no t_max change needed: wall_max (8 min - 2 min margin) ends the run, MAXSEG=1 prevents the chain
  RUNS="$runs" MARGIN=120 MAXSEG=1 sbatch --export=ALL --job-name=hst-seed-test \
      --account=$HST_ACCOUNT --partition=${HST_TEST_PARTITION:-$HST_PARTITION} $opts --time=00:08:00 horeka2_group.slurm
  echo "after it: ./check.sh logs test;  ./status.sh test"
  exit 0
fi

# every run starts in an empty directory: a Dati.cart.out without a log of ours would be read instead of the random field
for d in runs/*/; do
  if [ -f $d/Dati.cart.out ] && ! ls $d/run-*.log >/dev/null 2>&1; then
    echo "$d has a Dati.cart.out but no run log: a leftover would replace the random field. Remove it."; exit 1
  fi
done
runs=$(ls -d runs/*/ | sed 's#/$##' | tr '\n' ' ')
RUNS="$runs" sbatch --export=ALL --job-name=hst-seed \
    --account=$HST_ACCOUNT --partition=$HST_PARTITION $opts --time=$TIME horeka2_group.slurm
echo "squeue -u \$USER;  ./status.sh"
