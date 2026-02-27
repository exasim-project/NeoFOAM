# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(cmake/CPM.cmake)

if(NeoFOAM_BUILD_TESTS OR NeoFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(NAME Catch2 GITHUB_REPOSITORY catchorg/Catch2 VERSION 3.4.0)
endif()

if(NEOFOAM_BUILD_BINDINGS)
  cpmaddpackage(
    NAME
    nanobind
    GITHUB_REPOSITORY
    wjakob/nanobind
    GIT_TAG
    v2.10.2
    SYSTEM
    YES)
endif()

if(NEOFOAM_NEON_VIA_CPM)
  if(NOT DEFINED NEOFOAM_NEON_VERSION)
    set(NEOFOAM_NEON_VERSION
        "main"
        CACHE INTERNAL "")
  endif()

  cpmaddpackage(
    NAME
    NeoN
    GITHUB_REPOSITORY
    exasim-project/NeoN
    GIT_TAG
    ${NEOFOAM_NEON_VERSION}
    SYSTEM
    YES
    "Kokkos_ENABLE_CUDA ${Kokkos_ENABLE_CUDA}"
    "Kokkos_ENABLE_HIP ${Kokkos_ENABLE_HIP}")
endif()
