# Environment for HoreKA2 (KIT), the EasyBuild/Lmod software stack that
# replaces HoreKA (env/horeka.sh).
# Usage:  source env/horeka2.sh gpu [h100|h200|b200]    then   make GPU=1 NCCL=1
#         source env/horeka2.sh cpu                     then   make
#
# HoreKA2 has H100 and H200 (both cc90) and B200 (cc100) GPUs, no A100.
# GPU_ARCH, in order of precedence:
#   1. the device name given as second argument (use it on a login node, where
#      nvidia-smi sees no GPU), e.g.  source env/horeka2.sh gpu b200
#   2. an `export GPU_ARCH=...` made beforehand
#   3. detection with nvidia-smi when sourced on a GPU node
# Binaries are per architecture: keep them apart with BUILD, e.g.
#   source env/horeka2.sh gpu b200;  make GPU=1 NCCL=1 BUILD=build-b200
#
# Module lists can be overridden when sourcing, e.g.
#   HST_GPU_MODULES=NVHPC/26.9-CUDA-13.3.0 source env/horeka2.sh gpu
#   HST_CPU_MODULES="foss/2025b FFTW/3.3.10-GCC-14.3.0" source env/horeka2.sh cpu
#
# GPU: the code uses mpi_f08, so MPI must be built with the same Fortran
# compiler as the code (nvfortran).  The mpif90 is taken from
#   1. the loaded modules, if it wraps nvfortran, else
#   2. the HPC-X OpenMPI bundled in the NVHPC SDK tree ($NVHPC_ROOT/comm_libs/mpi),
# and the script stops with a description of what it found if there is none.
# The default is the toolchain module NVHPC/25.9-CUDA-12.9.1.  The first run on
# HoreKA2 showed that nvidia-compilers/25.9-CUDA-12.9.1 has no comm_libs/mpi.
# Alternative from the module list, H100/H200 only (CUDA 12.6 cannot target
# B200; nvfortran 24.9 has not been tried with this code):
#   HST_GPU_MODULES="NVHPC/24.9-CUDA-12.6.0 OpenMPI/5.0.3-NVHPC-24.9-CUDA-12.6.0"

HST_GPU_MODULES=${HST_GPU_MODULES:-NVHPC/25.9-CUDA-12.9.1}
# CPU: the same generation as the old gnu/13 + openmpi/5.0 + fftw/3.3_serial
# (GCC 13.3.0, OpenMPI 5.0.3, serial FFTW 3.3.10).
HST_CPU_MODULES=${HST_CPU_MODULES:-"GCC/13.3.0 OpenMPI/5.0.3-GCC-13.3.0 FFTW/3.3.10-GCC-13.3.0"}

case "${1:-gpu}" in
  gpu)
    # Checked before anything is purged or loaded, so a typo changes nothing.
    _hst_dev=${2:-}; _hst_dev=${_hst_dev,,}
    case "$_hst_dev" in
      "")         ;;
      h100|h200)  GPU_ARCH=cc90 ;;
      b200)       GPU_ARCH=cc100 ;;
      *)          echo "horeka2.sh: unknown device '$2' (h100, h200 or b200)" >&2; return 1 ;;
    esac
    module purge
    unset OPAL_PREFIX      # a stale one from an earlier `source` would point the other MPI at the wrong prefix
    module load $HST_GPU_MODULES || return 1

    # The EasyBuild module does not provide NVHPC_ROOT (the SDK version
    # directory that the job scripts use to find nsys/ncu); derive it from the
    # compiler: <root>/compilers/bin/nvfortran.
    _hst_nvf=$(command -v nvfortran) || { echo "horeka2.sh: nvfortran not found after loading $HST_GPU_MODULES" >&2; return 1; }
    export NVHPC_ROOT=$(dirname "$(dirname "$(dirname "$(readlink -f "$_hst_nvf")")")")

    # mpif90 must wrap nvfortran: the modules' own, else the SDK's HPC-X.
    _hst_wraps=$(mpif90 --showme:command 2>/dev/null) || _hst_wraps=
    if [[ $_hst_wraps != nvfortran && $_hst_wraps != */nvfortran && -x $NVHPC_ROOT/comm_libs/mpi/bin/mpif90 ]]; then
      export PATH=$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
      _hst_wraps=$(mpif90 --showme:command 2>/dev/null) || _hst_wraps=
    fi
    if [[ $_hst_wraps != nvfortran && $_hst_wraps != */nvfortran ]]; then
      echo "horeka2.sh: no MPI for nvfortran after loading: $HST_GPU_MODULES" >&2
      echo "  mpif90 on PATH:   $(command -v mpif90 || echo none)${_hst_wraps:+  (wraps $_hst_wraps)}" >&2
      echo "  SDK bundled MPI:  $NVHPC_ROOT/comm_libs/mpi  $([ -d "$NVHPC_ROOT/comm_libs/mpi" ] && echo present || echo missing)" >&2
      echo "  The code needs mpi_f08 built by nvfortran; try another list, e.g. HST_GPU_MODULES=\"...\" (see the header of env/horeka2.sh)." >&2
      return 1
    fi
    # NVHPC's HPC-X OpenMPI is relocated.  Without its real prefix,
    # mpirun/srun can find the libraries but MPI_Init looks for the OpenMPI
    # help files under the uninstalled HPC-X build path.  A prefix set by an MPI
    # module is kept.
    _hst_mpi=$(command -v mpif90)
    export OPAL_PREFIX=${OPAL_PREFIX:-$(dirname "$(dirname "$_hst_mpi")")}
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
    # Left unset on purpose when it cannot be determined: make would then pick
    # its own default (cc86), which is wrong for every GPU here.
    if [ -n "${GPU_ARCH:-}" ]; then
      export GPU_ARCH
    else
      echo "horeka2.sh: GPU_ARCH not set and no GPU visible; use 'source env/horeka2.sh gpu h100|h200|b200' or make GPU_ARCH=cc90|cc100" >&2
    fi
    ;;
  cpu)
    module purge
    unset OPAL_PREFIX
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
    echo "usage: source env/horeka2.sh [gpu [h100|h200|b200] | cpu]" >&2
    return 1
    ;;
esac
unset _hst_dev _hst_nvf _hst_wraps _hst_mpi _hst_d _hst_cc
