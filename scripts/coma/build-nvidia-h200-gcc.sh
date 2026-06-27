#!/bin/bash

# Run from the repository root (this script lives in scripts/coma/): cmake --preset needs
# CMakePresets.json there, and -DNEOFOAM_NEON_DIR=../NeoN and the *.log outputs are relative to it.
cd "$(dirname "$0")/../.." || exit 1

module purge
module load gcc/13.3.0
module load cuda/13.1.1 
module load cmake
module load openmpi

#rm -rf build
#source /storage/home/greole/code/NeoFOAM/.venv/bin/activate
source $HOME/OpenFOAM/openfoam/etc/bashrc

cd $HOME/code/NeoFOAM

export PRESET=profiling
export NEON_DEVICE=nvidia_h200

cmake --preset $PRESET \
	-DNeoN_WITH_THREADS=OFF \
	-DNeoN_WITH_UMPIRE=ON \
       	-DNeoN_WITH_OMP=OFF \
       	-DNEOFOAM_WITH_MPI=ON \
	-DCMAKE_C_COMPILER=$(which gcc) \
	-DCMAKE_CXX_COMPILER=$(which g++) \
       	-DKokkos_ENABLE_HIP=OFF \
	-DNeoN_BUILD_PYTHON_BINDINGS=OFF \
	-DNEOFOAM_BUILD_TESTS=ON \
	-DNEOFOAM_BUILD_BENCHMARKS=OFF \
	-DNEOFOAM_NEON_DIR=../NeoN \
       	> config-gcc-$NEON_DEVICE.log 2>&1

cmake --build   --preset $PRESET > build-$PRESET-gcc-$NEON_DEVICE.log 2>&1 
