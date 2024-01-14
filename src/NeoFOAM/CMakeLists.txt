# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM)

include(GNUInstallDirs)

target_include_directories(
  NeoFOAM PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                 "$<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>")

target_sources(NeoFOAM
               PRIVATE
               "core/Time.cpp"
               "core/Error.cpp"
               "core/Kokkos.cpp"
               "mesh/unstructuredMesh/unstructuredMesh.cpp"
               "cellCentredFiniteVolume/bcFields/scalar/fvccScalarFixedValueBoundaryField.cpp"
               "cellCentredFiniteVolume/grad/gaussGreenGrad.cpp")

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK
  NEOFOAM_ENABLE_SANITIZE_UB NEOFOAM_ENABLE_SANITIZE_THREAD
  NEOFOAM_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE Kokkos::kokkos)
