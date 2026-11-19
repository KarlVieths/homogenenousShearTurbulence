#!/bin/bash
# Why does a run see no OpenMP target device?  One node, one GPU, ~2 minutes.
#   cd <campaign directory> && sbatch --account=$HST_ACCOUNT --partition=$HST_PARTITION diag.slurm
# Output: logs/diag-<jobid>.out   (send it back as it is)
#SBATCH --job-name=hst-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=logs/diag-%j.out

HST_ROOT=${HST_ROOT:-$HOME/hst}
CAMP=$SLURM_SUBMIT_DIR
[ -r $HST_ROOT/env/horeka2.sh ] || { echo "no $HST_ROOT/env/horeka2.sh: submit with  HST_ROOT=<repository> sbatch ... diag.slurm"; exit 1; }
sec() { echo; echo "=== $* ==="; }

sec "node, Slurm"
hostname; echo "CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]  SLURM_JOB_GPUS=[$SLURM_JOB_GPUS]  GPU_DEVICE_ORDINAL=[$GPU_DEVICE_ORDINAL]"
env | grep -E "^(OMP_|NVCOMPILER|ACC_|CUDA_|UCX_)" | sort

sec "driver (the header line says the CUDA version the driver supports)"
nvidia-smi | head -4
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

sec "environment"
source $HST_ROOT/env/horeka2.sh gpu || echo "env/horeka2.sh failed"
echo "GPU_ARCH=$GPU_ARCH  NVHPC_ROOT=$NVHPC_ROOT"
module list 2>&1 | tail -8

gname=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr 'A-Z' 'a-z')
case "$gname" in *h200*) dev=h200;; *b200*) dev=b200;; *) dev=h100;; esac
[ -x $HST_ROOT/build-gpu-$dev/hst ] || dev=h100
EXE=$HST_ROOT/build-gpu-$dev/hst
echo "GPU '$gname' -> $EXE"
sec "libraries of $EXE (libcuda must resolve, not 'not found')"
ldd $EXE | grep -E "cuda|omp|nvc|not found"
echo "libcuda in ldconfig:"; ldconfig -p | grep libcuda

echo "LD_LIBRARY_PATH entries with 'stubs' (a stub libcuda hides every GPU):"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i stub || echo "  none"

sec "what the NVHPC runtime sees (nvaccelinfo)"
nvaccelinfo=$(command -v nvaccelinfo || ls $NVHPC_ROOT/compilers/bin/nvaccelinfo 2>/dev/null)
[ -n "$nvaccelinfo" ] && $nvaccelinfo -v 2>&1 | head -30 || echo "nvaccelinfo not found (NVHPC_ROOT=$NVHPC_ROOT)"

sec "device files and kernel modules (CUDA needs /dev/nvidia-uvm and /dev/nvidiactl, not only /dev/nvidia0)"
ls -l /dev/nvidia* 2>&1
grep -E "^nvidia" /proc/modules | cut -d' ' -f1,3
cat /proc/driver/nvidia-uvm/version 2>&1 | head -2

sec "fabric state (NVSwitch nodes need it 'Completed' before CUDA initialises)"
nvidia-smi -q -i 0 2>&1 | grep -i -A6 "fabric" | head -20
nvidia-smi topo -m 2>&1 | head -12
ls -l /dev/nvidia-caps 2>&1 | head -6
echo "libcuda_wrapper: $(ls -l /lib64/libcuda_wrapper.so.0 2>&1)"

sec "plain CUDA, no NVHPC: cudaGetDeviceCount with nvcc"
cat > $CAMP/logs/cudatest.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int n = -1; cudaError_t e = cudaGetDeviceCount(&n);
  printf("cudaGetDeviceCount = %d, error %d: %s\n", n, (int)e, cudaGetErrorString(e));
  int v = 0, d = 0; cudaDriverGetVersion(&d); cudaRuntimeGetVersion(&v);
  printf("driver supports %d, runtime %d\n", d, v);
  return 0;
}
EOF
if command -v nvcc >/dev/null; then
  nvcc -o $CAMP/logs/cudatest $CAMP/logs/cudatest.cu 2>&1 | tail -3 && $CAMP/logs/cudatest
else echo "no nvcc on PATH"; fi

sec "the binary, started three ways (60 s each at most; the log ends at the first step or the error)"
d=$CAMP/diag; rm -rf $d; mkdir -p $d; cp $CAMP/runs/smooth_kc2.09/theta_00.0/hst.in $d/; cd $d
echo "--- (a) as the campaign job starts it: CUDA_VISIBLE_DEVICES=0"
CUDA_VISIBLE_DEVICES=0 timeout 60 mpirun --bind-to none -np 1 $EXE hst.in 2>&1 | grep -E "ERROR|WARNING|s/step|device|rror" | head -5
echo "--- (b) with the Slurm value untouched: CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]"
timeout 60 mpirun --bind-to none -np 1 $EXE hst.in 2>&1 | grep -E "ERROR|WARNING|s/step|device|rror" | head -5
echo "--- (c) no mpirun, the binary directly, Slurm's value"
timeout 60 $EXE hst.in 2>&1 | grep -E "ERROR|WARNING|s/step|device|rror" | head -5
