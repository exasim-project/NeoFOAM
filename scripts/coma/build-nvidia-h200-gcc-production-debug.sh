#!/bin/bash
#
# Debuggable production build: identical to build-nvidia-h200-gcc-production.sh
# (CMAKE_BUILD_TYPE=Release, -O3 -DNDEBUG) so it reproduces the production
# SIGSEGV, but adds host -g / device -lineinfo so cuda-gdb and compute-sanitizer
# can symbolize the fault. CMAKE_CUDA_ARCHITECTURES is pinned to 90 (H200 = sm_90)
# so this can also be CONFIGURED/BUILT on a GPU-less login node (the base preset's
# "native" needs a local GPU to autodetect).

# Run from the repository root (this script lives in scripts/coma/): cmake --preset needs
# CMakePresets.json there, and -DNEOFOAM_NEON_DIR=../NeoN and the *.log outputs are relative to it.
cd "$(dirname "$0")/../.." || exit 1

module purge
module load gcc/13.3.0
module load cuda/12.8.1
module load cmake
module load openmpi

source $HOME/OpenFOAM/openfoam/etc/bashrc

export NEON_DEVICE=nvidia_h200
export PRESET=production-debug

cmake --preset $PRESET \
	-DCMAKE_CUDA_ARCHITECTURES=90 \
	-DKokkos_ARCH_HOPPER90=ON \
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
	> config-$PRESET-gcc-$NEON_DEVICE.log 2>&1

cmake --build --preset $PRESET > build-$PRESET-gcc-$NEON_DEVICE.log 2>&1
