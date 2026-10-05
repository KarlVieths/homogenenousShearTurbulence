#!/bin/bash
# Run four independent single-GPU simulations concurrently on one HoreKA node.
# Submit from a directory containing sim1/, ..., sim4/, each with hst.in.
#SBATCH --job-name=hst4
#SBATCH --account=hk-project-exasim
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=hst-%j.out

set -u

HST_ROOT=${HST_ROOT:-$HOME/hst}
HST_EXE=${HST_EXE:-$HST_ROOT/build-gpu-a100/hst}
source "$HST_ROOT/env/horeka.sh" gpu

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "MPI launcher: $(command -v mpirun)"
echo "OPAL_PREFIX=${OPAL_PREFIX:-unset}"
nvidia-smi -L

run_sim() {
    local dir=$1
    echo "Starting $dir"
    # The job allocation contains four Slurm tasks.  This single-task srun
    # step gets one GPU and one CPU allocation; --exclusive prevents overlap.
    # hst is MPI-enabled, so launch its one rank through the matching OpenMPI
    # rather than executing the MPI binary directly under srun.  --mpi=none
    # prevents Slurm from trying to provide a different MPI wireup layer.
    srun --exclusive --exact --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gpus-per-task=1 --gpu-bind=closest --mpi=none \
        mpirun -np 1 --bind-to none "$HST_EXE" "$dir/hst.in" >"$dir/hst.out" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        echo "$dir finished successfully"
    else
        echo "$dir FAILED with exit code $rc"
    fi
    return "$rc"
}

pids=()
for dir in sim1 sim2 sim3 sim4; do
    run_sim "$dir" &
    pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        status=1
    fi
done
exit "$status"