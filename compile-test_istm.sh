#!/usr/bin/env bash

make clean

##############################
# COMPILE AND TEST FOR CPU
##############################
# make				# CPU: gfortran + MPI + FFTW  -> build-cpu/hst
# make test
# tests/run_tests.sh build-cpu 2  # second argument: ranks

##############################
# COMPILE AND TEST FOR GPU
# (single-GPU nodes)
##############################
# source env/istm.sh              # istmio2 / istmcetus / istmcorax
# make GPU=1                      # GPU: nvfortran + cuFFT       -> build-gpu/hst
# make test GPU=1                 # the test programs, same build directory
# tests/run_tests.sh build-gpu 1  # second argument: ranks

##############################
# COMPILE AND TEST FOR GPU
# (mul,ti-GPU nodes)
##############################
source env/istm.sh           	# istmio2 / istmcetus / istmcorax
make GPU=1 NCCL=1          	    # the same with NCCL for the alltoall (multi-GPU nodes)
make test GPU=1 NCCL=1          # the test programs, same build directory; NCCL=1 keeps the test link of hst_mpi.o (-DHAVE_NCCL) consistent
tests/run_tests.sh build-gpu 2  # second argument: ranks
