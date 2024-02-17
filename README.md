**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Documentation](https://exasim-project.com/NeoFOAM/)** |
# NeoFOAM

## Requirements

NeoFOAM has the following requirements

*  _cmake > 3.22_
*  _gcc > 8.5_ or  _clang > 14_ 
*  _Kokkos 4.2.0_ (Preferably preinstalled, otherwise cloned and build at compile time) 

For NVIDIA GPU support
* cuda 

For building the documentation further dependencies like doxygen and sphinx are requirement. The list of requirements can be found [here](https://github.com/exasim-project/NeoFOAM/actions/workflows/doc.yml)


## Compilation

to install and download kokkos execute:

```bash 
    install_kokkos.sh
```



[![Build NeoFOAM](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml/badge.svg)](https://github.com/exasim-project/NeoFOAM/actions/workflows/build.yaml)
[![Gitter](https://img.shields.io/badge/Gitter-8A2BE2)](https://matrix.to/#/#NeoFOAM:gitter.im)

NeoFOAM uses cmake to build, thus the standard cmake procedure should work 

```bash 
    build.sh
```
