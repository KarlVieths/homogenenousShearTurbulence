#!/usr/bin/env bash

##############################
# COMPILE AND TEST FOR CPU
##############################
#make				# CPU: gfortran + MPI + FFTW  -> build-cpu/hst
#make test
#tests/run_tests.sh build-cpu 2  # second argument: ranks

##############################
# COMPILE AND TEST FOR GPU
##############################
source env/istm.sh           	 # istmio2 / istmcetus / istmcorax
# make GPU=1                   	 # GPU: nvfortran + cuFFT       -> build-gpu/hst
make GPU=1 NCCL=1          	 # the same with NCCL for the alltoall (multi-GPU nodes)
make test GPU=1              	 # the test programs, same build directory
tests/run_tests.sh build-gpu 2   # second argument: ranks
