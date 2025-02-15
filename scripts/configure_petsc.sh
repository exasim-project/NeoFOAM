# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 NeoFOAM authors


#!/bin/bash

## example script how to download, configure, compile and install petsc
## see also https://petsc.org/release/install/

CURRENTDIR="$(pwd)"

# clone latest release
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc

# configure petsc
./configure --with-64-bit-indices=0 --with-precision=double --with-cuda=1 --with-cuda-dir=/usr/local/cuda-12.3 --with-mpi=1 --with-fc=0 --force -use-gpu-aware-mpi=0 --with-mpi=1 --with-32bits-pci-domain=1 --prefix=opt/petsc --with-cc=mpicc --with-cxx=mpicxx --with-debugging=no --download-kokkos --download-kokkos-kernels --with-kokkos-kernels=1 --with-kokkos=1

# compile petsc
make PETSC_DIR=$CURRENTDIR/petsc PETSC_ARCH=arch-linux-c-opt all

#install petsc
make PETSC_DIR=$CURRENTDIR/petsc PETSC_ARCH=arch-linux-c-opt install

export PKG_CONFIG_PATH=$CURRENTDIR/opt/petsc/lib/pkgconfig
