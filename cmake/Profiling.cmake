# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
# NeoFOAM Profiling Integration of Kokkos Tools

include_guard(GLOBAL)

option(NEOFOAM_ENABLE_PROFILING
       "Enable profiling via Kokkos Tools"
       OFF)

if(NOT NEOFOAM_ENABLE_PROFILING)
  return()
endif()

message(STATUS "NeoFOAM profiling enabled")

# Force Kokkos to enable libdl support
set(Kokkos_ENABLE_LIBDL ON CACHE BOOL
    "Enable libdl support for Kokkos (required for tools)"
    FORCE)

# Fetch Kokkos Tools
include(FetchContent)

FetchContent_Declare(
  kokkos_tools
  GIT_REPOSITORY https://github.com/kokkos/kokkos-tools.git
  GIT_TAG master
)

FetchContent_MakeAvailable(kokkos_tools)

# Check that the top-level CMakeLists.txt exists
if(NOT EXISTS ${kokkos_tools_SOURCE_DIR}/CMakeLists.txt)
  message(FATAL_ERROR
    "Could not find the top-level CMakeLists.txt in kokkos-tools")
endif()

# -------------------------------
# Add kokkos-tools top-level CMakeLists.txt
# -------------------------------
add_subdirectory(
  ${kokkos_tools_SOURCE_DIR}                # source dir
  ${CMAKE_BINARY_DIR}/kokkos_tools_build    # build dir
)

# -------------------------------
# Set path to simpleKernelTimer library
# -------------------------------
# The target name is assumed to be 'kp_kernel_timer' in kokkos-tools
# CMake will place it in CMAKE_LIBRARY_OUTPUT_DIRECTORY
set(KOKKOS_TOOLS_LIB_PATH
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libkp_kernel_timer.so
)

if(NOT EXISTS ${KOKKOS_TOOLS_LIB_PATH})
  message(FATAL_ERROR
    "Could not find Kokkos simpleKernelTimer shared library at "
    "${KOKKOS_TOOLS_LIB_PATH}")
endif()

message(STATUS "Kokkos Tools library: ${KOKKOS_TOOLS_LIB_PATH}")

# -------------------------------
# Generate runtime environment helper script
# -------------------------------
file(WRITE
  ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh
  "export KOKKOS_TOOLS_LIBS=${KOKKOS_TOOLS_LIB_PATH}\n"
)

message(STATUS
  "Generated profiling env script: "
  "${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh")

message(STATUS
  "To enable profiling at runtime:\n"
  "  source ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh")
