# Environment for HoreKA2 (KIT), the EasyBuild/Lmod software stack that
# replaces HoreKA (env/horeka.sh).
# Usage:  source env/horeka2.sh gpu    then   make GPU=1 GPU_ARCH=ccXX NCCL=1
#         source env/horeka2.sh cpu    then   make
#
# GPU_ARCH is detected with nvidia-smi when the script is sourced on a GPU node
# (an `export GPU_ARCH=...` made beforehand is kept).  On a login node without
# GPUs it stays unset: export it or pass it to make yourself (cc80 A100,
# cc90 H100, cc100 B200, ...).
#
# Module versions can be overridden when sourcing, e.g.
#   HST_NVHPC_MODULE=nvidia-compilers/26.5-CUDA-13.2.0 source env/horeka2.sh gpu
#   HST_CPU_MODULES="foss/2025b FFTW/3.3.10-GCC-14.3.0" source env/horeka2.sh cpu
#
# GPU: the code uses mpi_f08, so MPI must be built with the same Fortran
# compiler as the code.  EasyBuild's OpenMPI builds are GCC (or NVHPC 24.9)
# builds and do not fit nvfortran 25.9; as on HoreKA and on the ISTM machines we
# use NVHPC's own HPC-X OpenMPI (CUDA-aware) from the SDK tree.

HST_NVHPC_MODULE=${HST_NVHPC_MODULE:-nvidia-compilers/25.9-CUDA-12.9.1}
# CPU: the same generation as the old gnu/13 + openmpi/5.0 + fftw/3.3_serial
# (GCC 13.3.0, OpenMPI 5.0.3, serial FFTW 3.3.10).
HST_CPU_MODULES=${HST_CPU_MODULES:-"GCC/13.3.0 OpenMPI/5.0.3-GCC-13.3.0 FFTW/3.3.10-GCC-13.3.0"}

case "${1:-gpu}" in
  gpu)
    module purge
    module load $HST_NVHPC_MODULE || return 1

    # The EasyBuild module does not provide NVHPC_ROOT (the SDK version
    # directory that the job scripts use to find nsys/ncu); derive it from the
    # compiler: <root>/compilers/bin/nvfortran.
    _hst_nvf=$(command -v nvfortran) || { echo "horeka2.sh: nvfortran not found after loading $HST_NVHPC_MODULE" >&2; return 1; }
    export NVHPC_ROOT=$(dirname "$(dirname "$(dirname "$(readlink -f "$_hst_nvf")")")")
    if [ ! -x "$NVHPC_ROOT/comm_libs/mpi/bin/mpif90" ]; then
      echo "horeka2.sh: no bundled MPI at $NVHPC_ROOT/comm_libs/mpi (is NVHPC_ROOT right?)" >&2
      return 1
    fi
    export PATH=$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
    # NVHPC ships a relocated HPC-X OpenMPI.  Without its real prefix,
    # mpirun/srun can find the libraries but MPI_Init looks for the OpenMPI
    # help files under the uninstalled HPC-X build path.
    export OPAL_PREFIX=$NVHPC_ROOT/comm_libs/mpi
    export UCX_MEMTYPE_CACHE=n
    # The EasyBuild NVHPC module does not necessarily put the SDK's runtime and
    # cuFFT directories on the library path (as env/istm.sh does by hand).
    for _hst_d in compilers/lib math_libs/lib64; do
      [ -d "$NVHPC_ROOT/$_hst_d" ] && export LD_LIBRARY_PATH=$NVHPC_ROOT/$_hst_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    done

    if [ -z "${GPU_ARCH:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
      _hst_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r')
      [ -n "$_hst_cc" ] && GPU_ARCH=cc$_hst_cc
    fi
    # Exported so that make sees it (also when it was only given as a prefix
    # assignment to `source`).  Left unset on purpose when it cannot be
    # detected: make would then pick its own default (cc86), which is wrong
    # for every GPU here.
    if [ -n "${GPU_ARCH:-}" ]; then
      export GPU_ARCH
    else
      echo "horeka2.sh: GPU_ARCH not set and no GPU visible; pass make GPU_ARCH=ccXX" >&2
    fi
    ;;
  cpu)
    module purge
    module load $HST_CPU_MODULES || return 1
    # EasyBuild sets EBROOTFFTW; the Makefile wants include/fftw3.f03 and lib/libfftw3 under FFTW_DIR.
    if [ -z "${EBROOTFFTW:-}" ]; then
      echo "horeka2.sh: EBROOTFFTW not set: no FFTW module in HST_CPU_MODULES ($HST_CPU_MODULES)" >&2
      return 1
    fi
    export FFTW_DIR=$EBROOTFFTW
    if [ ! -r "$FFTW_DIR/include/fftw3.f03" ] || ! compgen -G "$FFTW_DIR/lib/libfftw3.*" >/dev/null; then
      echo "horeka2.sh: $FFTW_DIR lacks include/fftw3.f03 or lib/libfftw3; set FFTW_DIR by hand" >&2
    fi
    ;;
  *)
    echo "usage: source env/horeka2.sh [gpu|cpu]" >&2
    return 1
    ;;
esac
unset _hst_nvf _hst_d _hst_cc
