Getting started
===============

You can build FoamAdapter by following these steps:

Clone the FoamAdapter repository including FoamAdapter integrated as a submodule:

   .. code-block:: bash

      git clone --recurse-submodules https://github.com/exasim-project/FoamAdapter.git

Navigate to the FoamAdapter directory:

   .. code-block:: bash

      cd FoamAdapter

FoamAdapter uses CMake to build, thus the standard CMake procedure should work, however, we recommend using one of the provided CMake presets detailed below `below <Building with CMake Presets>`_. From a build directory, you can execute:

   .. code-block:: bash

        mkdir build
        cd build
        cmake <DesiredBuildFlags> ..
        cmake --build .
        cmake --install .

Building with CMake Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, we provide several CMake presets to set commonly required flags.

   .. code-block:: bash

    cmake --list-presets # To list existing presets

To build FoamAdapter for production use, you can use the following commands:

   .. code-block:: bash

    cmake --preset production # To configure with ninja and common kokkos flags
    cmake --build --preset production # To compile with ninja and common kokkos flags

It should be noted that the build directory changes depending on the chosen preset. This way you can have different build directories for different presets and easily switch between them.

Prerequisites
^^^^^^^^^^^^^

The following tools are used in the development of this project:

The required tools for documentation:

.. code-block:: bash

    sudo apt install doxygen
    pip install pre-commit sphinx furo breathe sphinx-sitemap


The required tools for compilation (ubuntu latest 24.04):

.. code-block:: bash

    sudo apt update
    sudo apt install \
    ninja-build \
    clang-16 \
    gcc-10 \
    libomp-16-dev \
    python3 \
    python3-dev \
    build-essential

Run test case
^^^^^^^^^^^^^

To build these test cases, the CMake preset `profiling` should be used during config and build step for FoamAdapter.

.. code-block:: bash

   cmake --preset profiling
   cmake --build --preset profiling

Then go to the `tutorials` directory and use the predefined script `Allrun` in the directory of each test case to run the chosen test case.