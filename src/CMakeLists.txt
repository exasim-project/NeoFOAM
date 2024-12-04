# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM SHARED)

include(GNUInstallDirs)

target_sources(
  NeoFOAM
  PRIVATE "core/primitives/vector.cpp"
          "core/time.cpp"
          "core/database/database.cpp"
          "core/database/collection.cpp"
          "core/database/document.cpp"
          "core/database/fieldCollection.cpp"
          "core/database/oldTimeCollection.cpp"
          "core/dictionary.cpp"
          "core/demangle.cpp"
          "core/tokenList.cpp"
          "executor/CPUExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/serialExecutor.cpp"
          "mesh/unstructured/boundaryMesh.cpp"
          "mesh/unstructured/unstructuredMesh.cpp"
          "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
          "finiteVolume/cellCentred/stencil/geometryScheme.cpp"
          "finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp"
          "finiteVolume/cellCentred/boundary/boundary.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenGrad.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenDiv.cpp"
          "finiteVolume/cellCentred/interpolation/linear.cpp"
          "finiteVolume/cellCentred/interpolation/upwind.cpp")

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  target_sources(NeoFOAM PRIVATE "core/mpi/halfDuplexCommBuffer.cpp"
                                 "mesh/unstructured/communicator.cpp")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK NEOFOAM_ENABLE_SANITIZE_UB
  NEOFOAM_ENABLE_SANITIZE_THREAD NEOFOAM_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE neofoam_warnings neofoam_options)
target_link_libraries(NeoFOAM PUBLIC NeoFOAM_public_api Kokkos::kokkos cpptrace::cpptrace)
if(NEOFOAM_ENABLE_MPI_SUPPORT)
  target_link_libraries(NeoFOAM PUBLIC MPI::MPI_CXX)
endif()
