**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Documentation](https://exasim-project.com/NeoFOAM/latest)** |
# NeoFOAM

> [!IMPORTANT]
> The NeoFOAM project needs you!
> If you're interested in contributing to NeoFOAM please open a PR! If you have any questions on where to start please contact us here or on [gitter](https://matrix.to/#/#NeoFOAM:gitter.im).

## Requirements

NeoFOAM has the following requirements

*  _cmake > 3.22_
*  _gcc >= 10_ or  _clang >= 16_
*  _Kokkos 4.3.0_

For NVIDIA GPU support
* cuda

For development it is required to use [pre-commit](https://pre-commit.com/).

### C++ dependencies

The cmake build process will prefer system wide installed C++ dependencies like Kokkos, cxxopts, etc. If you prefer to clone, configure and build dependencies your self consider setting `-DCPM_USE_LOCAL_PACKAGES = OFF`, see [CPM](https://github.com/cpm-cmake/CPM.cmake) for more details.

### Documentation build

For building the documentation further dependencies like doxygen and sphinx are requirement. The list of requirements can be found [here](https://github.com/exasim-project/NeoFOAM/actions/workflows/doc.yml)

## Compilation procedure

[![Build NeoFOAM](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml/badge.svg)](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml)
[![Gitter](https://img.shields.io/badge/Gitter-8A2BE2)](https://matrix.to/#/#NeoFOAM:gitter.im)

NeoFOAM uses cmake to build, thus the standard cmake procedure should work. From a build directory you can execute

    cmake <DesiredBuildFlags> ..
    cmake --build .
    cmake --install .

Additionally, we provide several Cmake presets to set commmonly required flags if you compile NeoFoam in combination with Kokkos.

    cmake --list-presets # To list existing presets
    cmake --preset ninja-kokkos-cuda # To compile with ninja and common kokkos flags for CUDA devices



## Executing Benchmarks

NeoFOAM provides a set of benchmarks which can be executed and plotted by the following commands

    cmake --build . --target execute_benchmarks # runs the benchmark suite
    cmake --build . --target execute_execute_plot_benchmark # plots the benchmark results


## Executing Tests

    cmake --build . --target test
