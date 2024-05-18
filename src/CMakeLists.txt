# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
# add_subdirectory(DSL)

add_library(NeoFOAM)
include(GNUInstallDirs)

target_include_directories(
  NeoFOAM PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                 "$<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>")

target_sources(
  NeoFOAM
  PRIVATE "core/Time.cpp"
          "core/Dictionary.cpp"
          "core/Kokkos.cpp"
          "executor/OMPExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/CPUExecutor.cpp"
          "mesh/unstructuredMesh/UnstructuredMesh.cpp"
          "mesh/unstructuredMesh/BoundaryMesh.cpp"
          "mesh/stencil/StencilDataBase.cpp")

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK
  NEOFOAM_ENABLE_SANITIZE_UB NEOFOAM_ENABLE_SANITIZE_THREAD
  NEOFOAM_ENABLE_SANITIZE_MEMORY)

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoFOAM INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoFOAM INTERFACE NF_DEBUG_INFO)
endif()

target_link_libraries(NeoFOAM PRIVATE Kokkos::kokkos cpptrace::cpptrace)
