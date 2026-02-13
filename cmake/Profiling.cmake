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
# User can specify kokkos-tools source (optional)
# -------------------------------
set(KOKKOS_TOOLS_SRC_DIR "" CACHE PATH
    "Path to kokkos-tools source directory. If empty, will fetch from GitHub.")

if(NOT KOKKOS_TOOLS_SRC_DIR)
  include(FetchContent)
  FetchContent_Declare(
    kokkos_tools
    GIT_REPOSITORY https://github.com/kokkos/kokkos-tools.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(kokkos_tools)
  set(KOKKOS_TOOLS_SRC_DIR ${kokkos_tools_SOURCE_DIR})
endif()

if(NOT EXISTS ${KOKKOS_TOOLS_SRC_DIR})
  message(FATAL_ERROR "kokkos-tools source directory does not exist: ${KOKKOS_TOOLS_SRC_DIR}")
endif()

# -------------------------------
# Build directory for Kokkos Tools
# -------------------------------
set(KOKKOS_TOOLS_BUILD_DIR "${CMAKE_BINARY_DIR}/kokkos_tools_build")
file(MAKE_DIRECTORY ${KOKKOS_TOOLS_BUILD_DIR})

# -------------------------------
# Define the library path
# -------------------------------
set(KOKKOS_TOOLS_LIB_PATH "${KOKKOS_TOOLS_BUILD_DIR}/profiling/simple-kernel-timer/libkp_kernel_timer.so")

# -------------------------------
# Custom command to configure and build kokkos-tools
# -------------------------------
add_custom_command(
  OUTPUT ${KOKKOS_TOOLS_LIB_PATH} ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh
  COMMAND ${CMAKE_COMMAND} -S ${KOKKOS_TOOLS_SRC_DIR} -B ${KOKKOS_TOOLS_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${KOKKOS_TOOLS_BUILD_DIR} -- -j$<NUMBER_OF_PROCESSORS>
  COMMAND ${CMAKE_COMMAND} -E echo "export KOKKOS_TOOLS_LIBS=${KOKKOS_TOOLS_LIB_PATH}" > ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building Kokkos Tools and generating kokkos_profiling_env.sh"
  VERBATIM
)

# -------------------------------
# Custom target that depends on the above command
# -------------------------------
add_custom_target(build_kokkos_tools ALL
  DEPENDS ${KOKKOS_TOOLS_LIB_PATH} ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh
  COMMENT "Automatic build of Kokkos Tools for profiling"
)

message(STATUS "Profiling setup: Kokkos Tools will be built automatically after NeoFOAM.")
message(STATUS "Library path: ${KOKKOS_TOOLS_LIB_PATH}")
message(STATUS "Environment script: ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh")
