**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
# NeoFOAM

## Requirements

NeoFOAM has the following requirements

*  _cmake 3.28+_
*  _clang 17+_ 
* cuda

## Compilation

to install and download kokkos execute:

```bash 
    install_kokkos.sh
```



[![Build NeoFOAM](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml/badge.svg)](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml)

NeoFOAM uses cmake to build, thus the standard cmake procedure should work 

```bash 
    build.sh
```