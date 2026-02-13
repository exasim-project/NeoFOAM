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

# --- Build kokkos-tools the same as manual steps ---

# Build directory inside the kokkos-tools root
set(KOKKOS_TOOLS_BUILD_DIR "${kokkos_tools_SOURCE_DIR}/build")
file(MAKE_DIRECTORY ${KOKKOS_TOOLS_BUILD_DIR})

# Add the top-level CMakeLists.txt in this build directory
add_subdirectory(
  ${kokkos_tools_SOURCE_DIR}   # source dir
  ${KOKKOS_TOOLS_BUILD_DIR}    # build dir
)

# --- Locate the resulting shared library ---
# For simpleKernelTimer, the shared library is under build/profiling/simple-kernel-timer
set(KOKKOS_TOOLS_LIB_PATH
    ${KOKKOS_TOOLS_BUILD_DIR}/profiling/simple-kernel-timer/libkp_kernel_timer.so
    CACHE INTERNAL "Path to Kokkos kernel timer library")

if(NOT EXISTS ${KOKKOS_TOOLS_LIB_PATH})
  message(WARNING "Kokkos kernel timer library not found: ${KOKKOS_TOOLS_LIB_PATH}")
endif()

message(STATUS "Kokkos Tools library: ${KOKKOS_TOOLS_LIB_PATH}")

# Generate runtime environment helper script
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
