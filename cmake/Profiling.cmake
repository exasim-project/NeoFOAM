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

# Build simple-kernel-timer tool
set(KOKKOS_KERNEL_TIMER_SOURCE
    ${kokkos_tools_SOURCE_DIR}/profiling/simple-kernel-timer)

if(NOT EXISTS ${KOKKOS_KERNEL_TIMER_SOURCE}/CMakeLists.txt)
  message(FATAL_ERROR
    "Could not find simple-kernel-timer in kokkos-tools")
endif()

add_subdirectory(
  ${KOKKOS_KERNEL_TIMER_SOURCE}
  ${CMAKE_BINARY_DIR}/kokkos_kernel_timer
)

# Define path to built tool
set(KOKKOS_TOOLS_LIB_PATH
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libkp_kernel_timer.so
    CACHE INTERNAL "Path to Kokkos kernel timer library")

message(STATUS "Kokkos Tools library: ${KOKKOS_TOOLS_LIB_PATH}")

# Generate runtime environment helper script
file(WRITE
  ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh
  "export KOKKOS_TOOLS_LIBS=${KOKKOS_TOOLS_LIB_PATH}\n"
)

message(STATUS
  "Generated profiling env script: "
  "${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh")

# Optional: Print runtime hint
message(STATUS
  "To enable profiling at runtime:\n"
  "  source ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh")
