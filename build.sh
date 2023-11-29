#!/bin/bash


# Run CMake to generate the build files
# cmake -G "Ninja" -S . -B build  -DKokkos_DIR="$PWD/Kokkos/lib/cmake/Kokkos" -DCMAKE_BUILD_TYPE=Debug
cmake -G "Ninja" -S . -B build  -DKokkos_DIR="$PWD/Kokkos/lib/cmake/Kokkos" -DCMAKE_BUILD_TYPE=Release

# Build the project using make
cmake --build build
