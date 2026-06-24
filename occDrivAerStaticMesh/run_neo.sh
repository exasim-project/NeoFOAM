#!/bin/bash
# Per-rank GPU binding: each MPI rank sees exactly one GPU (its local rank).
# OpenFOAM parses argv before Kokkos init, so we cannot use --kokkos device
# flags; bind via CUDA_VISIBLE_DEVICES instead.
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
exec /storage/home/greole/code/NeoFOAM/build/profilingnvidia_h100/bin/neoSimpleFoam -parallel
