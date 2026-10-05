#!/usr/bin/env bash

set -euo pipefail

# make clean

##############################
# COMPILE AND TEST FOR CPU
##############################
# make				# CPU: gfortran + MPI + FFTW  -> build-cpu/hst
# make test
# tests/run_tests.sh build-cpu 2  # second argument: ranks

##############################
# COMPILE AND TEST FOR GPU
# (single-GPU nodes)
source env/istm.sh             		# ISTM-volans / istmio2 / istmcetus / istmcorax
BUILD=${BUILD:-build-gpu-istmvolans}
RANKS=${RANKS:-1}

# Use exactly the same BUILD directory for compilation and testing.  The
# previous script compiled into build-gpu-istmvolans but ran the tests from
# build-gpu, which either tested an old executable or failed if build-gpu did
# not exist.
make GPU=1 BUILD="$BUILD" 		# GPU: nvfortran + cuFFT -> $BUILD/hst
make test GPU=1 BUILD="$BUILD"		# the test programs, same build directory
tests/run_tests.sh "$BUILD" "$RANKS" 		# second argument: ranks

##############################
# COMPILE AND TEST FOR GPU
# (multi-GPU nodes)
##############################
# source env/istm.sh           	# istmio2 / istmcetus / istmcorax
# make GPU=1 NCCL=1          	# the same with NCCL for the alltoall (multi-GPU nodes)
# make test GPU=1 NCCL=1          # the test programs, same build directory; NCCL=1 keeps the test link of hst_mpi.o (-DHAVE_NCCL) consistent
# tests/run_tests.sh build-gpu 2  # second argument: ranks
