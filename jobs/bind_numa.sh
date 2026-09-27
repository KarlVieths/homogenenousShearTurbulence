#!/bin/bash
# mpirun wrapper: run the rank on the cores and memory of the NUMA domain
# of its GPU (the GPU is the local rank modulo the number of GPUs, as in
# select_device), so that the host side of NCCL sits next to the GPU and
# its InfiniBand adapter.   mpirun ... jobs/bind_numa.sh build-gpu/hst hst.in
lr=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}
ndev=$(nvidia-smi -L | wc -l)
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $((lr % ndev)) | tr 'A-F' 'a-f' | sed 's/^0000//')
numa=$(cat /sys/bus/pci/devices/$bus/numa_node 2>/dev/null)
[ "${numa:--1}" -ge 0 ] || numa=0
[ "$lr" -eq 0 ] && echo "   bind_numa: rank 0 on GPU $bus, NUMA node $numa"
exec numactl --cpunodebind=$numa --membind=$numa "$@"
