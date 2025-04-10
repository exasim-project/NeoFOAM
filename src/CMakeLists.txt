# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM ${NeoFOAM_LIB_TYPE})

include(GNUInstallDirs)

if(NEOFOAM_ENABLE_CUDA)
  set_source_files_properties(${NeoFOAM_SRCS} PROPERTIES LANGUAGE CUDA)
endif()
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
          "dsl/coeff.cpp"
          "dsl/explicit.cpp"
          "dsl/spatialOperator.cpp"
          "dsl/temporalOperator.cpp"
          "executor/CPUExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/serialExecutor.cpp"
          "linearAlgebra/utilities.cpp"
          "mesh/unstructured/boundaryMesh.cpp"
          "mesh/unstructured/unstructuredMesh.cpp"
          "finiteVolume/cellCentred/linearAlgebra/sparsityPattern.cpp"
          "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
          "finiteVolume/cellCentred/stencil/geometryScheme.cpp"
          "finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp"
          "finiteVolume/cellCentred/stencil/cellToFaceStencil.cpp"
          "finiteVolume/cellCentred/boundary/boundary.cpp"
          "finiteVolume/cellCentred/operators/ddtOperator.cpp"
          # "finiteVolume/cellCentred/operators/expression.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenGrad.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenDiv.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp"
          "finiteVolume/cellCentred/operators/sourceTerm.cpp"
          "finiteVolume/cellCentred/operators/surfaceIntegrate.cpp"
          "finiteVolume/cellCentred/interpolation/linear.cpp"
          "finiteVolume/cellCentred/interpolation/upwind.cpp"
          "finiteVolume/cellCentred/faceNormalGradient/uncorrected.cpp"
          "finiteVolume/cellCentred/auxiliary/coNum.cpp"
          "timeIntegration/timeIntegration.cpp"
          "timeIntegration/rungeKutta.cpp")

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  target_sources(NeoFOAM PRIVATE "core/mpi/halfDuplexCommBuffer.cpp"
                                 "mesh/unstructured/communicator.cpp")
endif()

if(NEOFOAM_WITH_GINKGO)
  target_sources(NeoFOAM PRIVATE "linearAlgebra/ginkgo.cpp")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK NEOFOAM_ENABLE_SANITIZE_UB
  NEOFOAM_ENABLE_SANITIZE_THREAD NEOFOAM_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE neofoam_warnings neofoam_options)
target_link_libraries(NeoFOAM PUBLIC NeoFOAM_public_api Kokkos::kokkos sundials_core
                                     sundials_arkode sundials_nvecserial cpptrace::cpptrace)

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
