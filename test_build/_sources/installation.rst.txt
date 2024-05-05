Installation
============

You can install NeoFOAM by following these steps:

Clone the NeoFOAM repository:

   .. code-block:: bash

      git clone https://github.com/exasim-project/NeoFOAM.git

Navigate to the NeoFOAM directory:

   .. code-block:: bash

      cd NeoFOAM

NeoFOAM uses Cmake to build, thus the standard Cmake procedure should work, however, we recommend using one of the provided Cmake presets detailed below `below <Building with Cmake Presets>`_. From a build directory, you can execute:

   .. code-block:: bash

        mkdir build
        cd build
        cmake <DesiredBuildFlags> ..
        cmake --build .
        cmake --install .

The following can be chained with -D<DesiredBuildFlags>=<Value> to the Cmake command most and most relevant build flags are: 

+---------------------------+-----------------------------------+---------+
| Flag                      | Description                       | Default |
+===========================+===================================+=========+
| CMAKE_BUILD_TYPE          | Build in debug or release mode    | Debug   |
+---------------------------+-----------------------------------+---------+
| NEOFOAM_BUILD_APPS        | Build NeoFOAM with Applications   | ON      |
+---------------------------+-----------------------------------+---------+
| NEOFOAM_BUILD_BENCHMARKS  | Build NeoFOAM with benchmarks     | OFF     |
+---------------------------+-----------------------------------+---------+
| NEOFOAM_BUILD_DOC         | Build NeoFOAM with documentation  | ON      |
+---------------------------+-----------------------------------+---------+
| NEOFOAM_BUILD_TESTS       | Build NeoFOAM with tests          | OFF     |
+---------------------------+-----------------------------------+---------+
| Kokkos_ENABLE_SERIAL      | Enable Serial backend for Kokkos  | ON      |
+---------------------------+-----------------------------------+---------+
| Kokkos_ENABLE_OPENMP      | Enable OpenMP backend for Kokkos  | OFF     |
+---------------------------+-----------------------------------+---------+
| Kokkos_ENABLE_ROCM        | Enable ROCm backend for Kokkos    | OFF     |
+---------------------------+-----------------------------------+---------+
| Kokkos_ENABLE_SYCL        | Enable SYCL backend for Kokkos    | OFF     |
+---------------------------+-----------------------------------+---------+
| Kokkos_ENABLE_CUDA        | Enable CUDA backend for Kokkos    | OFF     |
+---------------------------+-----------------------------------+---------+

By opening the the project with cmake-gui you can easily set these flags and configure the build.

Building with Cmake Presets
^^^^^^^^^^^^^^^^^^^^^

Additionally, we provide several Cmake presets to set commonly required flags if you compile NeoFoam in combination with Kokkos.

   .. code-block:: bash

    cmake --list-presets # To list existing presets

To build NeoFOAM with Kokkos and CUDA support, you can use the following commands:

   .. code-block:: bash 

    cmake --preset ninja-kokkos-cuda # To configure with ninja and common kokkos flags for CUDA devices
    cmake --build --preset ninja-kokkos-cuda # To compile with ninja and common kokkos flags for CUDA devices

It should be noted that the build directory changes depending on the chosen preset. This way you can have different build directories for different presets and easily switch between them.
