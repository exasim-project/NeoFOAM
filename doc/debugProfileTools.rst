Using Kokkos Tools
======================
Kokkos provides a powerful tools interface for debugging and profiling applications.
This section outlines how to use Kokkos tools effectively in the context of NeoFOAM development.

----------------------------------
Kokkos Tools available in NeoFOAM
----------------------------------
NeoFOAM integrates several Kokkos tools to assist developers in optimizing performance and diagnosing issues.

**Kokkos Profiling Tools**: For performance analysis and optimization.
- simple-kernel-timer
- space-time-stack
- memory-high-water-mark

**Kokkos Debugging Tools**: For identifying bugs in Kokkos-based code.
- kernel-logger

----------------------------------
Setting Up and Using Kokkos Tools
----------------------------------
**For debugging:**

1. Configure and build NeoFOAM with the ``develop`` preset.
2. Run your application with the kernel logger enabled::

   .. code-block:: bash

      ./run_with_kokkos_tool.sh debug kernel-logger [--log filename] ./your_neofoam_application

**For profiling:**

1. Configure and build NeoFOAM with the ``profiling`` preset.
2. Run your application with the desired profiling tool::

   .. code-block:: bash

      ./run_with_kokkos_tool.sh profile <available Kokkos profiling tool> [--log filename] ./your_neofoam_application

**Note**: The ``--log`` option allows you to specify a file to save the tool's output for later analysis.
