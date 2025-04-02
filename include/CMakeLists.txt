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

if(NEOFOAM_WITH_GINKGO)
  target_link_libraries(NeoFOAM_public_api INTERFACE Ginkgo::ginkgo)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_GINKGO=1)
  # set_property( TARGET Ginkgo::ginkgo APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS )
elseif(NEOFOAM_WITH_PETSC)
  add_definitions(${PETSc_DEFINITIONS})
  target_link_libraries(NeoFOAM_public_api INTERFACE ${PETSc_LIBRARIES})
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_PETSC=1)
else()
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_GINKGO=0)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_PETSC=0)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_DEBUG_INFO)
endif()
