# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# NeoFOAM Profiling Integration via Kokkos Tools
#
include_guard(GLOBAL)

if(NOT KOKKOS_TOOLS_ENABLE)
  return()
endif()

message(STATUS "Kokkos Tools enabled")

# ---------------------------------------------------------------------------
# Ensure Kokkos supports runtime tool loading
# ---------------------------------------------------------------------------

set(Kokkos_ENABLE_LIBDL
    ON
    CACHE BOOL "Enable libdl support for Kokkos (required for tools)" FORCE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

set(KOKKOS_TOOLS_GIT_TAG
    33693813b125096cc72a90d495fd02a7bb7574ea
    CACHE STRING "Git tag or commit for kokkos-tools")

set(KOKKOS_TOOLS_BUILD_DIR ${CMAKE_BINARY_DIR}/kokkos_tools_build)

include(ExternalProject)

# ---------------------------------------------------------------------------
# ExternalProject definition (no install step)
# ---------------------------------------------------------------------------

message(STATUS "Cloning kokkos-tools (tag: ${KOKKOS_TOOLS_GIT_TAG})")

ExternalProject_Add(
  kokkos_tools_ext
  GIT_REPOSITORY https://github.com/kokkos/kokkos-tools.git
  GIT_TAG ${KOKKOS_TOOLS_GIT_TAG}
  SOURCE_DIR ${CMAKE_BINARY_DIR}/kokkos_tools_src
  BINARY_DIR ${KOKKOS_TOOLS_BUILD_DIR}
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON
             -DKokkosTools_ENABLE_MPI=OFF
             -DKokkosTools_ENABLE_PAPI=OFF
             -DKokkosTools_ENABLE_CALIPER=OFF
             -DKokkosTools_ENABLE_APEX=OFF
             -DKokkosTools_ENABLE_EXAMPLES=OFF
             -DKokkosTools_ENABLE_TESTS=OFF
  INSTALL_COMMAND ""
  USES_TERMINAL_BUILD TRUE)

# ---------------------------------------------------------------------------
# Build target
# ---------------------------------------------------------------------------

add_custom_target(
  build_kokkos_tools ALL
  DEPENDS kokkos_tools_ext
  COMMENT "Building Kokkos Tools")

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

message(STATUS "--------------------------------------------------")
message(STATUS "Kokkos Tools build dir : ${KOKKOS_TOOLS_BUILD_DIR}")
message(STATUS "--------------------------------------------------")
