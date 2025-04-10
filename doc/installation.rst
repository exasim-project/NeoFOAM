Installation
============

You can build NeoN by following these steps:

Clone the NeoN repository:

   .. code-block:: bash

      git clone https://github.com/exasim-project/NeoN.git

Navigate to the NeoN directory:

   .. code-block:: bash

      cd NeoN

NeoN uses CMake to build, thus the standard CMake procedure should work, however, we recommend using one of the provided CMake presets detailed below `below <Building with CMake Presets>`_. From a build directory, you can execute:

   .. code-block:: bash

        mkdir build
        cd build
        cmake <DesiredBuildFlags> ..
        cmake --build .
        cmake --install .

The following can be chained with -D<DesiredBuildFlags>=<Value> to the CMake command most and most relevant build flags are:

+---------------------------+-----------------------------------+---------+
| Flag                      | Description                       | Default |
+===========================+===================================+=========+
| CMAKE_BUILD_TYPE          | Build in debug or release mode    | Debug   |
+---------------------------+-----------------------------------+---------+
| NeoN_BUILD_DOC         | Build NeoN with documentation  | ON      |
+---------------------------+-----------------------------------+---------+
| NeoN_BUILD_TESTS       | Build NeoN with tests          | OFF     |
+---------------------------+-----------------------------------+---------+

To browse the full list of build options it is recommended to use a build tool like ``ccmake``.
By opening the the project with cmake-gui you can easily set these flags and configure the build.
NeoN specific build flags are prefixed by ``NeoN_``.

.. note::

   NeoN will automatically enable ``Kokkos_ENABLE_CUDA`` or ``Kokkos_ENABLE_HIP`` if either of this is available on
   the system. This can be prevented by setting both options explicitly.

Building with CMake Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, we provide several CMake presets to set commonly required flags if you compile NeoN in combination with Kokkos.

   .. code-block:: bash

    cmake --list-presets # To list existing presets

To build NeoN for production use, you can use the following commands:

   .. code-block:: bash

    cmake --preset production # To configure with ninja and common kokkos flags
    cmake --build --preset production # To compile with ninja and common kokkos flags

It should be noted that the build directory changes depending on the chosen preset. This way you can have different build directories for different presets and easily switch between them.

Prerequisites
^^^^^^^^^^^^^

The following tools are used in the development of this project:

required tools for documentation:

.. code-block:: bash

    sudo apt install doxygen
    pip install pre-commit sphinx furo breathe sphinx-sitemap


required tools for compilation (ubuntu latest 24.04):

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


Workflow with vscode
^^^^^^^^^^^^^^^^^^^^

install the following extensions:

.. code-block:: bash

   ms-vscode.cpptools
   ms-vscode.cmake-tools


After installation, you can open the NeoN directory with vscode and configure the build with cmake presets with the cmake extension as shown below:

.. figure:: _static/installation/cmakePresets.gif
   :alt: configure the build with cmake presets
   :align: center

After configuring the build, you can build the project with the build button or test in "testing" tab (flask icon).

To create the documentation, you can use the 'Build Sphinx Documentation' task in the vscode task menu. Type `Ctrl+P` and type `task` and press space and the build documentation and press enter. The documentation will be created in the `docs_build` directory.
