# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
# add_subdirectory(DSL)

add_library(NeoFOAM SHARED)

include(GNUInstallDirs)

target_sources(
  NeoFOAM
  PRIVATE "core/mpi/halfDuplexCommBuffer.cpp"
          "core/primitives/vector.cpp"
          "core/time.cpp"
          "core/dictionary.cpp"
          "core/kokkos.cpp"
          "executor/OMPExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/CPUExecutor.cpp"
          "mesh/unstructured/boundaryMesh.cpp"
          "mesh/unstructured/communicator.cpp"
          "mesh/unstructured/unstructuredMesh.cpp"
          "mesh/stencil/stencilDataBase.cpp"
          "mesh/stencil/stencilDataBase.cpp"
          "finiteVolume/boundary.cpp")

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK NEOFOAM_ENABLE_SANITIZE_UB
  NEOFOAM_ENABLE_SANITIZE_THREAD NEOFOAM_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE neofoam_warnings neofoam_options)
target_link_libraries(NeoFOAM PUBLIC NeoFOAM_public_api MPI::MPI_CXX Kokkos::kokkos
                                     cpptrace::cpptrace)
