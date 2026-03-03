Using Kokkos Tools
==================

Kokkos provides a powerful tools interface for debugging and profiling
applications. This section explains how to use Kokkos tools effectively
in NeoFOAM development.

Kokkos Tools Available in NeoFOAM
----------------------------------

NeoFOAM integrates several Kokkos tools to assist developers in
optimizing performance and diagnosing issues.

**Profiling tools**

- ``simple-kernel-timer``
- ``space-time-stack``
- ``memory-high-water-mark``

**Debugging tools**

- ``kernel-logger``

Setting Up and Using Kokkos Tools
----------------------------------

For debugging
^^^^^^^^^^^^^

1. Configure and build NeoFOAM using the ``develop`` preset::

      cmake --preset develop
      cmake --build --preset develop

2. Run your application with the kernel logger enabled::

      ./run_with_kokkos_tool.sh debug kernel-logger [--log filename] ./your_neofoam_application

For profiling
^^^^^^^^^^^^^

1. Configure and build NeoFOAM using the ``profiling`` preset::

      cmake --preset profiling
      cmake --build --preset profiling

2. Run your application with the desired profiling tool::

      ./run_with_kokkos_tool.sh profile <tool-name> [--log filename] ./your_neofoam_application

   where ``<tool-name>`` is one of:

   - ``simple-kernel-timer``
   - ``space-time-stack``
   - ``memory-high-water-mark``

.. note::

   The ``--log`` option writes the tool output to a file for later analysis.
