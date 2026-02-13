# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# NeoFOAM Profiling Integration via Kokkos Tools
#

include_guard(GLOBAL)

option(NEOFOAM_ENABLE_PROFILING
       "Enable profiling via Kokkos Tools"
       OFF)

if(NOT NEOFOAM_ENABLE_PROFILING)
  return()
endif()

message(STATUS "NeoFOAM profiling enabled")

# ---------------------------------------------------------------------------
# Kokkos must support dynamic tool loading
# ---------------------------------------------------------------------------

set(Kokkos_ENABLE_LIBDL ON CACHE BOOL
    "Enable libdl support for Kokkos (required for tools)"
    FORCE)

# ---------------------------------------------------------------------------
# Acquire kokkos-tools source
# ---------------------------------------------------------------------------

set(KOKKOS_TOOLS_SRC_DIR "" CACHE PATH
    "Path to existing kokkos-tools source directory (optional)")

if(NOT KOKKOS_TOOLS_SRC_DIR)

  include(FetchContent)

  # Pin a commit for CI reproducibility
  set(KOKKOS_TOOLS_GIT_TAG  master CACHE STRING
      "33693813b125096cc72a90d495fd02a7bb7574ea")

  FetchContent_Declare(
    kokkos_tools
    GIT_REPOSITORY https://github.com/kokkos/kokkos-tools.git
    GIT_TAG        ${KOKKOS_TOOLS_GIT_TAG}
  )

  FetchContent_GetProperties(kokkos_tools)

  if(NOT kokkos_tools_POPULATED)
    FetchContent_Populate(kokkos_tools)
  endif()

  set(KOKKOS_TOOLS_SRC_DIR ${kokkos_tools_SOURCE_DIR})

endif()

if(NOT EXISTS ${KOKKOS_TOOLS_SRC_DIR}/CMakeLists.txt)
  message(FATAL_ERROR
    "Invalid kokkos-tools source directory: ${KOKKOS_TOOLS_SRC_DIR}")
endif()

# ---------------------------------------------------------------------------
# Build kokkos-tools as external project
# ---------------------------------------------------------------------------

include(ExternalProject)

set(KOKKOS_TOOLS_BUILD_DIR
    ${CMAKE_BINARY_DIR}/kokkos_tools_build)

set(KOKKOS_TOOLS_INSTALL_DIR
    ${CMAKE_BINARY_DIR}/kokkos_tools_install)

ExternalProject_Add(kokkos_tools_ext

  SOURCE_DIR ${KOKKOS_TOOLS_SRC_DIR}
  BINARY_DIR ${KOKKOS_TOOLS_BUILD_DIR}

  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${KOKKOS_TOOLS_INSTALL_DIR}
    -DBUILD_SHARED_LIBS=ON
    -DKokkosTools_ENABLE_MPI=OFF
    -DKokkosTools_ENABLE_PAPI=OFF
    -DKokkosTools_ENABLE_CALIPER=OFF
    -DKokkosTools_ENABLE_APEX=OFF
    -DKokkosTools_ENABLE_EXAMPLES=OFF
    -DKokkosTools_ENABLE_TESTS=OFF

  INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
  USES_TERMINAL_BUILD TRUE
)

# ---------------------------------------------------------------------------
# Define expected tool library
# ---------------------------------------------------------------------------

set(KOKKOS_TOOLS_LIB_PATH
    ${KOKKOS_TOOLS_INSTALL_DIR}/lib/libkp_kernel_timer.so)

# ---------------------------------------------------------------------------
# Generate runtime environment script
# ---------------------------------------------------------------------------

add_custom_target(build_kokkos_tools ALL
  DEPENDS kokkos_tools_ext
  COMMENT "Building Kokkos Tools for NeoFOAM profiling"
)

add_custom_command(
  TARGET build_kokkos_tools
  POST_BUILD

  COMMAND ${CMAKE_COMMAND} -E echo
          "export KOKKOS_TOOLS_LIBS=${KOKKOS_TOOLS_LIB_PATH}"
          > ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh

  COMMENT "Generating kokkos_profiling_env.sh"
)

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------

message(STATUS "Kokkos Tools source:  ${KOKKOS_TOOLS_SRC_DIR}")
message(STATUS "Kokkos Tools build:   ${KOKKOS_TOOLS_BUILD_DIR}")
message(STATUS "Kokkos Tools install: ${KOKKOS_TOOLS_INSTALL_DIR}")
message(STATUS "Profiling env script: ${CMAKE_BINARY_DIR}/kokkos_profiling_env.sh")
