# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM_public_api INTERFACE) # dummy target

set_property(
  TARGET NeoFOAM_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set_property(
  TARGET NeoFOAM_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")

if(NEOFOAM_ENABLE_GINKGO)
  target_link_libraries(NeoFOAM_public_api INTERFACE Ginkgo::ginkgo)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_GINKGO=1)
else()
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_GINKGO=1)
endif()

if(NEOFOAM_ENABLE_PETSC)
  target_link_libraries(NeoFOAM_public_api INTERFACE PkgConfig::PETSc)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_PETSC=1)
else()
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_PETSC=1)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_DEBUG_INFO)
endif()
