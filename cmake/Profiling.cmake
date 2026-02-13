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

# -------------------------------
# Fetch Kokkos Tools
# -------------------------------
include(FetchContent)

FetchContent_Declare(
  kokkos_tools
  GIT_REPOSITORY https://github.com/kokkos/kokkos-tools.git
  GIT_TAG master
)

FetchContent_GetProperties(kokkos_tools)
if(NOT kokkos_tools_POPULATED)
  FetchContent_Populate(kokkos_tools)
endif()

set(KOKKOS_TOOLS_BUILD_DIR "${kokkos_tools_SOURCE_DIR}/build")
file(MAKE_DIRECTORY ${KOKKOS_TOOLS_BUILD_DIR})

# -------------------------------
# Build kokkos-tools
# -------------------------------
message(STATUS "Configuring Kokkos Tools...")
execute_process(
  COMMAND ${CMAKE_COMMAND} ..
  WORKING_DIRECTORY ${KOKKOS_TOOLS_BUILD_DIR}
  RESULT_VARIABLE kokkos_cmake_result
  OUTPUT_VARIABLE kokkos_cmake_out
  ERROR_VARIABLE kokkos_cmake_err
)
if(NOT kokkos_cmake_result EQUAL 0)
  message(FATAL_ERROR "Failed to configure Kokkos Tools:\n${kokkos_cmake_err}")
endif()

message(STATUS "Building Kokkos Tools...")
execute_process(
  COMMAND ${CMAKE_COMMAND} --build . -- -j$(nproc)
  WORKING_DIRECTORY ${KOKKOS_TOOLS_BUILD_DIR}
  RESULT_VARIABLE kokkos_build_result
  OUTPUT_VARIABLE kokkos_build_out
  ERROR_VARIABLE kokkos_build_err
)
if(NOT kokkos_build_result EQUAL 0)
  message(FATAL_ERROR "Failed to build Kokkos Tools:\n${kokkos_build_err}")
endif()

# -------------------------------
# Locate the simpleKernelTimer shared library
# -------------------------------
set(KOKKOS_TOOLS_LIB_PATH
    "${KOKKOS_TOOLS_BUILD_DIR}/profiling/simple-kernel-timer/libkp_kernel_timer.so"
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
