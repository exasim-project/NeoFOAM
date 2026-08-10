**[Requirements](#requirements)** |
**[Compilation](#Compilation)** |
**[Documentation](https://exasim-project.com/NeoFOAM/develop)** |
# NeoFOAM

**NeoFOAM** is an open-source **performance-portable finite-volume CFD framework** for 
modern heterogeneous computing systems. Using [NeoN](https://github.com/exasim-project/NeoN) as its computational backend, NeoFOAM enables the same CFD solver implementation to execute across serial CPUs, multithreaded CPUs, MPI-based distributed systems, and GPUs from NVIDIA, AMD, and Intel.

The framework is designed for portability across **Linux, macOS, and Windows** 
while providing efficient execution on both CPU and GPU architectures.

NeoFOAM includes fluid-flow solvers that correspond to established OpenFOAM solvers. 
For example, `neoIcoFoam` is the NeoFOAM equivalent of **OpenFOAM** `icoFoam` for transient 
incompressible laminar flow simulations of Newtonian fluids.

To ensure numerical reproducibility, NeoFOAM solvers are validated against their 
OpenFOAM counterparts, allowing users to compare results directly while benefiting 
from a modern performance-portable implementation.

## Key Features 

- Performance-portable finite-volume infrastructure built on top of NeoN 
- Execution on serial CPU, multithreaded CPU, MPI-based distributed systems, and GPUs from NVIDIA, AMD, and Intel 
- Portability across Linux, macOS, and Windows 
- OpenFOAM-compatible mesh and field conversion utilities 
- OpenFOAM-equivalent solver implementations (e.g., `neoIcoFoam`) 
- Validation against OpenFOAM or literature results for numerical reproducibility

## Requirements

NeoFOAM has the following requirements

*  _cmake 3.22+_
*  _gcc >= 12_ or  _clang >= 18+_
* OpenFOAM _2406_+
* NeoN (latest version)

## Compilation

We provide several Cmake presets to set commmonly required flags for building NeoFOAM

    cmake --list-presets # List existing presets
    cmake --preset production # Configure for production
    cmake --build --preset production # Build for production

Check the documentation for [details](https://exasim-project.com/NeoFOAM/develop/gettingStarted.html#getting-started)

## Structure

The repository is structured in the following way:
- *src* and *include* implement common functionality to copy data between OpenFOAM and NeoN
- *tests* demonstrating that NeoFOAM and OpenFOAM deliver identical results are provided by this repository in the test folder.
- *examples* provides examples of how NeoFOAM can be used for writing applications
- *tutorials* provides tutorial cases which can be run like typical OpenFOAM cases

## Agentic coding

We provide an `AGENTS.md` file to support LLM-based coding workflows.
Please instruct AI coding tools to reference this file, for example by prompting: “See `AGENTS.md` for shared project instructions.”
