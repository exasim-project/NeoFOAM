**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
# NeoFOAM

## Requirements

NeoFOAM has the following requirements

*  _cmake 3.28+_
*  _clang 17+_ 
*  _Kokkos 4.2.0_ (Preferably preinstalled, otherwise cloned and build at compile time) 

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

# build documentation

```bash 
    #assume python and doxygen is installed
    pip install sphinx
    pip install sphinx-rtd-theme
    pip install breathe
    pip3 install sphinx-sitemap
    # 
    ./build_docs.sh
    firefox docs/_build/html/index.html # open index page in firefox
```