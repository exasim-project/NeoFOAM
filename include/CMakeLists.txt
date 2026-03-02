# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM_public_api INTERFACE) # dummy target

if(NeoN_WITH_SUNDIALS)
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_SUNDIALS=1)
else()
  target_compile_definitions(NeoFOAM_public_api INTERFACE NF_WITH_SUNDIALS=0)
endif()

set_property(
  TARGET NeoFOAM_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set_property(
  TARGET NeoFOAM_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(NeoFOAM_public_api INTERFACE NeoN::NeoN NeoN::NeoN_public_api)

# Get list of some *.hpp files in folder include
file(GLOB_RECURSE include_files *.hpp)

# Convert the list of files into #includes
foreach(include_file ${include_files})
  set(include_statements "${include_statements}#include \"${include_file}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/NeoFOAM/NeoFOAM.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/NeoFOAM/NeoFOAM.hpp)
