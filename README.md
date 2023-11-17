**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
# NeoFOAM

## Requirements

NeoFOAM has the following requirements

*  _cmake 3.28+_
*  _clang 17+_ 

## Compilation
![build](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yml/badge.svg)

NeoFOAM uses cmake to build, thus the standard cmake procedure should work 

    mkdir build && cd build && cmake ..
    cmake --build .
    cmake --install .
