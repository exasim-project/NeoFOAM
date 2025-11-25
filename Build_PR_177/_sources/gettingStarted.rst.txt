Getting started
===============

You can build NeoFOAM by following these steps:

Clone the NeoFOAM repository:

   .. code-block:: bash

      git clone https://github.com/exasim-project/NeoFOAM.git

Navigate to the NeoFOAM directory:

   .. code-block:: bash

      cd NeoFOAM

NeoFOAM uses CMake to build, thus the standard CMake procedure should work, however, we recommend using one of the provided CMake presets detailed below `below <Building with CMake Presets>`_. From a build directory, you can execute:

   .. code-block:: bash

        mkdir build
        cd build
        cmake <DesiredBuildFlags> ..
        cmake --build .
        cmake --install .

Build NeoFOAM against NeoN
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three ways to build NeoFOAM against NeoN:
1. Using the NeoN repo directory:

   We can specify the path to the NeoN repo directory during the CMake configuration step:
   .. code-block:: bash

      -DNEOFOAM_NEON_DIR=/path/to/NeoN/

2. Using the NeoN submodule:
   We can initialize and update the NeoN submodule in the NeoFOAM repo:
   .. code-block:: bash

      git submodule update --init --recursive

   Then, during the CMake configuration step, CMake will automatically detect and use the NeoN submodule.

3. Using automatically downloaded NeoN:
   During the CMake configuration step, if neither of the above two options are provided, CMake will automatically download a pre-defined version of NeoN.
   The pre-defined version is the main branch of NeoN by default, but it can be changed by specifying the desired version as follows:
   .. code-block:: bash

      -DNEOFOAM_NEON_VERSION=<desired_version>

   The desired version can be a branch name, a tag name, or a commit hash.

Building for GPUs
^^^^^^^^^^^^^^^^^^

NeoFOAM supports GPUs from different vendors through NeoN, which uses Kokkos as the backend for performance portability.
Check the NeoN documentation for [instructions](https://exasim-project.com/NeoN/latest/installation.html#building-for-gpus)
 on how to build for GPUs from different vendors.

Building with CMake Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, we provide several CMake presets to set commonly required flags.

   .. code-block:: bash

    cmake --list-presets # To list existing presets

To build NeoFOAM for production use, you can use the following commands:

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

To build these test cases, the CMake preset `profiling` should be used during config and build step for NeoFOAM.

.. code-block:: bash

   cmake --preset profiling
   cmake --build --preset profiling

Then go to the `tutorials` directory and use the predefined script `Allrun` in the directory of each test case to run the chosen test case.
