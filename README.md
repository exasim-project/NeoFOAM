**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Documentation](https://exasim-project.com/NeoFOAM/)** |
# NeoFOAM

## Requirements

NeoFOAM has the following requirements

*  _cmake 3.28+_
*  _clang 17+_ 
*  _Kokkos 4.2.0_ (Preferably preinstalled, otherwise cloned and build at compile time) 

For building the documentation further depencies like doxygen and sphinx are requirement. The list of requirements can be found [here](https://github.com/exasim-project/NeoFOAM/actions/workflows/doc.yml)


## Compilation

[![Build NeoFOAM](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml/badge.svg)](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml)

NeoFOAM uses cmake to build, thus the standard cmake procedure should work 

    mkdir build && cd build && cmake ..
    cmake --build .
    cmake --install .
