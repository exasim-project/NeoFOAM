**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Documentation](https://exasim-project.com/NeoFOAM/latest)** |

[![workflows/github-linux](https://github.com/exasim-project/neofoam/actions/workflows/build_on_ubuntu.yaml/badge.svg?branch=master)](https://github.com/exasim-project/neofoam/actions/workflows/build_on_ubuntu.yaml?query=branch%3Amaster)
[![workflows/github-OSX](https://github.com/exasim-project/neofoam/actions/workflows/build_on_macos.yaml/badge.svg?branch=master)](https://github.com/exasim-project/neofoam/actions/workflows/build_on_macos.yaml?query=branch%3Amaster)
[![workflows/github-windows](https://github.com/exasim-project/neofoam/actions/workflows/build_on_windows.yaml/badge.svg?branch=master)](https://github.com/exasim-project/neofoam/actions/workflows/build_on_windows.yaml?query=branch%3Amaster)

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
* cuda _12+_

For development it is required to use [pre-commit](https://pre-commit.com/).

### C++ dependencies

The cmake build process will prefer system wide installed C++ dependencies like Kokkos, cxxopts, etc. If you prefer to clone, configure and build dependencies your self consider setting `-DCPM_USE_LOCAL_PACKAGES = OFF`, see [CPM](https://github.com/cpm-cmake/CPM.cmake) for more details.

### Documentation build

For building the documentation further dependencies like doxygen and sphinx are requirement. The list of requirements can be found [here](https://github.com/exasim-project/NeoFOAM/actions/workflows/build_doc.yaml)

## Compilation procedure

[![Build NeoFOAM](https://github.com/exasim-project/NeoFOAM/actions/workflows/build_on_ubuntu.yaml/badge.svg)](https://github.com/exasim-project/NeoFOAM/actions/workflows/build_on_ubuntu.yaml)
[![Gitter](https://img.shields.io/badge/Gitter-8A2BE2)](https://matrix.to/#/#NeoFOAM:gitter.im)

NeoFOAM uses cmake to build, thus the standard cmake procedure should work. From a build directory you can execute

    cmake <DesiredBuildFlags> ..
    cmake --build .
    cmake --install .

Additionally, we provide several Cmake presets to set commmonly required flags if you compile NeoFoam in combination with Kokkos.

    cmake --list-presets # To list existing presets
    cmake --preset production # To compile for production use



## Executing Benchmarks

NeoFOAM provides a set of benchmarks which can be executed and plotted by the following commands

    cmake --build . --target execute_benchmarks # runs the benchmark suite
    cmake --build . --target execute_plot_benchmark # plots the benchmark results


## Executing Tests

    cmake --build . --target test
