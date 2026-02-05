# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

cmake_minimum_required(VERSION 3.22.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "An implementation of FOAM")

if(NOT DEFINED ENV{FOAM_SRC})
  message(FATAL_ERROR "You must source OpenFOAM before building NeoFOAM")
endif()
if(NOT $ENV{FOAM_API} GREATER_EQUAL 2406)
  message(WARNING "OpenFOAM version < 2406")
endif()

option(NEOFOAM_BUILD_EXAMPLES "Build the NeoN examples" ON)
option(NEOFOAM_BUILD_TESTS "Build the unit tests" OFF)
option(NEOFOAM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NEOFOAM_ENABLE_PROFILING
       "Enable profiling hooks (Kokkos Tools integration)"
       OFF)
set(NEOFOAM_BENCHMARK_MODE
    "all"
    CACHE STRING "perform only a subset of the benchmarks")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Add the cmake folder so the find_package command finds custom packages
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set the output directories for all binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(cmake/OpenFOAM.cmake)

if(NOT DEFINED NEOFOAM_NEON_DIR)
  if(EXISTS ${PROJECT_SOURCE_DIR}/src/NeoN/CMakeLists.txt AND NOT DEFINED NEOFOAM_NEON_VERSION)
    message(STATUS "src/NeoN/CMakeLists.txt found, using NeoN")
    add_subdirectory(src/NeoN)
  else()
    message(STATUS "src/NeoN/CMakeLists.txt not found, using CPM")
    set(NEOFOAM_NEON_VIA_CPM ON)
  endif()
else()
  message(STATUS "using NEOFOAM_NEON_DIR ${NEOFOAM_NEON_DIR}")
  add_subdirectory(${NEOFOAM_NEON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/NeoN)
endif()

include(cmake/CxxThirdParty.cmake)

add_subdirectory(include)
add_subdirectory(src)

if(NEOFOAM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(NEOFOAM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(NEOFOAM_BUILD_BENCHMARKS)
  enable_testing()
  add_subdirectory(benchmarks)
endif()

include(cmake/banner.cmake)
