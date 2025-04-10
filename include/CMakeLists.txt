# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoN authors

add_library(NeoN_public_api INTERFACE) # dummy target

set_property(
  TARGET NeoN_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set_property(
  TARGET NeoN_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")

if(NeoN_WITH_GINKGO)
  target_link_libraries(NeoN_public_api INTERFACE Ginkgo::ginkgo)
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_GINKGO=1)
  # set_property( TARGET Ginkgo::ginkgo APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS )
else()
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_GINKGO=0)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoN_public_api INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoN_public_api INTERFACE NF_DEBUG_INFO)
endif()

# Get list of some *.hpp files in folder include
file(GLOB_RECURSE include_files *.hpp)

# Convert the list of files into #includes
foreach(include_file ${include_files})
  set(include_statements "${include_statements}#include \"${include_file}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/NeoN/NeoN.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/NeoN/NeoN.hpp)
