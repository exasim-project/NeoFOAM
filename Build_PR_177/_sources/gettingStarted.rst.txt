Getting Started
===============

Follow these steps to build NeoFOAM:

Clone the NeoFOAM repository:

.. code-block:: bash

   git clone https://github.com/exasim-project/NeoFOAM.git

Navigate to the NeoFOAM directory:

.. code-block:: bash

   cd NeoFOAM

NeoFOAM uses CMake for building. The standard CMake procedure works, but we recommend using one of the provided CMake presets (see :ref:`cmake-presets`). From a build directory, execute:

.. code-block:: bash

     mkdir build
     cd build
     cmake <DesiredBuildFlags> ..
     cmake --build .
     cmake --install .

Build NeoFOAM Against NeoN
--------------------------

There are three ways to link NeoFOAM with NeoN:

1. **Using the NeoN repository directory**

   Specify the path to the NeoN repository during CMake configuration:

   .. code-block:: bash

      -DNEOFOAM_NEON_DIR=/path/to/NeoN/

2. **Using the NeoN submodule**

   Initialize and update the NeoN submodule in the NeoFOAM repository:

   .. code-block:: bash

      git submodule update --init --recursive

   During CMake configuration, the submodule will be automatically detected.

3. **Using automatically downloaded NeoN**

   If neither of the above options is provided, CMake will download a predefined version of NeoN. By default, this is the `main` branch. To specify a different version:

   .. code-block:: bash

      -DNEOFOAM_NEON_VERSION=<desired_version>

   `<desired_version>` can be a branch name, tag, or commit hash.

Building for GPUs
-----------------

NeoFOAM supports GPUs from multiple vendors via NeoN, which uses Kokkos for performance portability.
For detailed instructions, see the `NeoN GPU installation guide <https://exasim-project.com/NeoN/latest/installation.html#building-for-gpus>`_.

Building with CMake Presets
---------------------------

.. _cmake-presets:

Several CMake presets are provided to simplify building with common configurations:

.. code-block:: bash

   cmake --list-presets  # List all available presets

For production builds:

.. code-block:: bash

   cmake --preset production       # Configure with Ninja and common Kokkos flags
   cmake --build --preset production  # Compile with Ninja and common Kokkos flags

Each preset uses its own build directory, allowing multiple configurations in parallel.

Prerequisites
-------------

The following tools are required:

**Documentation tools:**

.. code-block:: bash

   sudo apt install doxygen
   pip install pre-commit sphinx furo breathe sphinx-sitemap

**Compilation tools (Ubuntu 24.04):**

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

Running Test Cases
------------------

To build test cases, use the `profiling` CMake preset:

.. code-block:: bash

   cmake --preset profiling
   cmake --build --preset profiling

Then navigate to the `tutorials` directory and run the provided `Allrun` script in the directory of the chosen test case:

.. code-block:: bash

   cd tutorials/<test_case>
   chmod +x Allrun  # Make sure it is executable
   ./Allrun
