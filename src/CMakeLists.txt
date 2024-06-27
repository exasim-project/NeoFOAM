# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
# add_subdirectory(DSL)

add_library(NeoFOAM)

set_property(TARGET NeoFOAM PROPERTY POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

target_sources(
  NeoFOAM
  PRIVATE "core/time.cpp"
          "core/dictionary.cpp"
          "core/kokkos.cpp"
          "executor/OMPExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/CPUExecutor.cpp"
          "mesh/unstructured/unstructuredMesh.cpp"
          "mesh/unstructured/boundaryMesh.cpp"
          "mesh/stencil/stencilDataBase.cpp"
          "finiteVolume/boundary.cpp")

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK NEOFOAM_ENABLE_SANITIZE_UB
  NEOFOAM_ENABLE_SANITIZE_THREAD NEOFOAM_ENABLE_SANITIZE_MEMORY)

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoFOAM INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoFOAM INTERFACE NF_DEBUG_INFO)
endif()

target_link_libraries(
  NeoFOAM
  PUBLIC NeoFOAM_public_api
  PRIVATE Kokkos::kokkos cpptrace::cpptrace)
