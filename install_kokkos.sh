#!/bin/bash
wget https://github.com/kokkos/kokkos/archive/refs/tags/4.2.00.zip
unzip 4.2.00.zip 
cmake -S kokkos-4.2.00 -B KokkosBuild \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=Kokkos \
 -DKokkos_ENABLE_OPENMP=ON \
 -DKokkos_ENABLE_CUDA=ON

cmake --build KokkosBuild --target install
