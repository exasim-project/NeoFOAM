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
          "core/demangle.cpp"
          "core/tokenList.cpp"
          "core/kokkos.cpp"
          "executor/CPUExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/serialExecutor.cpp"
          "mesh/unstructured/boundaryMesh.cpp"
          "mesh/unstructured/communicator.cpp"
          "mesh/unstructured/unstructuredMesh.cpp"
          "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
          "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
          "finiteVolume/cellCentred/stencil/geometryScheme.cpp"
          "finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp"
          "finiteVolume/cellCentred/fieldDatabase.cpp"
          "finiteVolume/cellCentred/solutionFields.cpp"
          "finiteVolume/cellCentred/boundary/boundary.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenGrad.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenDiv.cpp"
          "finiteVolume/cellCentred/interpolation/linear.cpp"
          "finiteVolume/cellCentred/interpolation/upwind.cpp")

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK NEOFOAM_ENABLE_SANITIZE_UB
  NEOFOAM_ENABLE_SANITIZE_THREAD NEOFOAM_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE neofoam_warnings neofoam_options)
target_link_libraries(NeoFOAM PUBLIC NeoFOAM_public_api MPI::MPI_CXX Kokkos::kokkos
                                     cpptrace::cpptrace)
