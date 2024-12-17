# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
# add_subdirectory(DSL)

add_library(NeoFOAM ${NeoFOAM_LIB_TYPE})

include(GNUInstallDirs)

set(NeoFOAM_SRCS
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
    "mesh/unstructured/unstructuredMesh.cpp"
    "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
    "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
    "finiteVolume/cellCentred/stencil/geometryScheme.cpp"
    "finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp"
    "finiteVolume/cellCentred/boundary/boundary.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenGrad.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenDiv.cpp"
    "finiteVolume/cellCentred/interpolation/linear.cpp"
    "finiteVolume/cellCentred/interpolation/upwind.cpp")

target_sources(NeoFOAM PRIVATE ${NeoFOAM_SRCS})

if(NEOFOAM_ENABLE_CUDA)
  set_source_files_properties(${NeoFOAM_SRCS} PROPERTIES LANGUAGE CUDA)
endif()
if(NEOFOAM_ENABLE_MPI_SUPPORT)
  target_sources(NeoFOAM PRIVATE "core/mpi/halfDuplexCommBuffer.cpp"
                                 "mesh/unstructured/communicator.cpp")
  if(NEOFOAM_ENABLE_CUDA)
    set_source_files_properties("core/mpi/halfDuplexCommBuffer.cpp"
                                "mesh/unstructured/communicator.cpp" PROPERTIES LANGUAGE CUDA)
  endif()
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
if(WIN32)
  set_target_properties(
    NeoFOAM
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/$<0:>)
endif()
