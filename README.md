**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Documentation](https://exasim-project.com/NeoFOAM/develop)** |
# NeoFOAM

NeoFOAM provides platform-portable implementations of common CFD algorithms and solvers using
[NeoN](https://github.com/exasim-project/NeoN) as a computational backend.
Additionally, It implements converters between OpenFOAM and NeoN datastructures such that you can run examples, benchmarks and tests on accelerator devices.

## Requirements

NeoFOAM has the following requirements

*  _cmake 3.22+_
*  _gcc >= 12_ or  _clang >= 18+_
* OpenFOAM _2406_+
* CUDA  _12.1_ (for GPU support)

## Compilation

We provide several Cmake presets to set commmonly required flags for building NeoFOAM

    cmake --list-presets # To list existing presets
    cmake --preset production # Config for production
    cmake --build --preset production # Build for production

## Structure

The repository is structured in the following way:
- *src* and *include* implement common functionality to copy data between OpenFOAM and NeoN
- *tests* demonstrating that NeoFOAM and OpenFOAM deliver identical results are provided by this repository in the test folder.
- *examples* provides examples of how NeoFOAM can be used for writing applications
- *tutorials* provides tutorial cases which can be run like typical OpenFOAM cases
