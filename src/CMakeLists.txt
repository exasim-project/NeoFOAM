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
  PRIVATE 
  "core/Time.cpp"
  "core/Dictionary.cpp"
  "core/Error.cpp"
  "core/Kokkos.cpp"

  "primitives/vector.cpp"
  
  "executor/OMPExecutor.cpp"
  "executor/GPUExecutor.cpp"
  "executor/CPUExecutor.cpp"
  "mesh/unstructuredMesh/unstructuredMesh.cpp"
  "mesh/unstructuredMesh/BoundaryMesh.cpp"
  "mesh/stencil/StencilDataBase.cpp"
  "mesh/stencil/FvccGeometryScheme.cpp"
  "mesh/stencil/BasicFvccGeometryScheme.cpp"
  
  "cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarFixedValueBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarZeroGradientBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarEmptyBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarCalculatedBoundaryField.cpp"
  
  "cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorFixedValueBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorZeroGradientBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorEmptyBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorCalculatedBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/fvccBoundaryFieldSelector.cpp"
  
  "cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarEmptyBoundaryField.cpp"
  "cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarCalculatedBoundaryField.cpp"

  "cellCentredFiniteVolume/grad/gaussGreenGrad.cpp"
  # "cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationFactory.cpp"
  "cellCentredFiniteVolume/surfaceInterpolation/linear.cpp")

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK
  NEOFOAM_ENABLE_SANITIZE_UB NEOFOAM_ENABLE_SANITIZE_THREAD
  NEOFOAM_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE Kokkos::kokkos)
