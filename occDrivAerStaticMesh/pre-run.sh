#!/bin/bash
# Per-rank GPU binding: each MPI rank sees exactly one GPU (its local rank).
# OpenFOAM parses argv before Kokkos init, so we cannot use --kokkos device
# flags; bind via CUDA_VISIBLE_DEVICES instead.


#!/bin/bash

module purge
module load gcc/13.3.0
module load cuda/13.0.2
module load cmake
module load openmpi

source $HOME/OpenFOAM/openfoam/etc/bashrc
export NEON_DEVICE=nvidia_h200
#export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
export CUDA_VISIBLE_DEVICES=1,2,3,4


echo; echo "potentialFoam"
time mpirun -np 4  potentialFoam -initialiseUBCs -parallel $fileHandler > 30_potentialFoam.log 2>&1 || exit 1

echo; echo "applyBoundaryLayer"
time mpirun -np 4 applyBoundaryLayer -ybl "0.0450244" -parallel $fileHandler > 40_applyBoundaryLayer.log 2>&1 || exit 1

