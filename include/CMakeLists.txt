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
else()
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_GINKGO=0)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_DEBUG_INFO)
endif()

if(NEOFOAM_DP_SCALAR)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NEOFOAM_DP_SCALAR=1)
endif()
if(NEOFOAM_DP_LABEL)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NEOFOAM_DP_LABEL=1)
endif()

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_MPI_SUPPORT=1)
  if(NEOFOAM_ENABLE_MPI_WITH_THREAD_SUPPORT)
    target_compile_definitions(NeoFOAM_public_api INTERFACE NF_REQUIRE_MPI_THREAD_SUPPORT=1)
  endif()
endif()

# Get list of some *.hpp files in folder include
file(GLOB_RECURSE include_files *.hpp)

# Convert the list of files into #includes
foreach(include_file ${include_files})
  set(include_statements "${include_statements}#include \"${include_file}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/NeoFOAM/NeoFOAM.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/NeoFOAM/NeoFOAM.hpp)
