# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")

find_package(MPI 3.1 REQUIRED)
find_package(Kokkos ${KOKKOS_CHECKOUT_VERSION} QUIET)

if(NOT ${Kokkos_FOUND})
  include(FetchContent)

  FetchContent_Declare(
    kokkos
    QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    GIT_TAG ${KOKKOS_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Kokkos)
endif()

include(cmake/CPM.cmake)

cpmaddpackage(NAME cpptrace GITHUB_REPOSITORY jeremy-rifkin/cpptrace VERSION
              0.5.4)

cpmaddpackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json VERSION 3.11.3)

cpmaddpackage(NAME spdlog GITHUB_REPOSITORY gabime/spdlog VERSION 1.13.0)

cpmaddpackage(NAME cxxopts GITHUB_REPOSITORY jarro2783/cxxopts VERSION 3.2.0)

if(NEOFOAM_BUILD_TESTS OR NEOFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(NAME Catch2 GITHUB_REPOSITORY catchorg/Catch2 VERSION 3.4.0)
endif()
