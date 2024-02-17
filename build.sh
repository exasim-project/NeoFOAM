#!/bin/bash


# Run CMake to generate the build files
# cmake -G "Ninja" -S . -B build  -DKokkos_DIR="$PWD/Kokkos/lib/cmake/Kokkos" -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build \
 -DCMAKE_BUILD_TYPE=Release \
 -DKokkos_ENABLE_SERIAL=ON \
 -DKokkos_ENABLE_OPENMP=ON \
 -DKokkos_ENABLE_CUDA=ON


# Build the project using make
cmake --build build -j
