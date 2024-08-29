# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(NEOFOAM_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NEOFOAM_KOKKOS_CHECKOUT_VERSION)

find_package(MPI 3.1 REQUIRED)
find_package(Kokkos ${NEOFOAM_KOKKOS_CHECKOUT_VERSION} QUIET)

if(NOT ${Kokkos_FOUND})
  include(FetchContent)

  include(cmake/AutoEnableDevice.cmake)

  FetchContent_Declare(
    kokkos
    SYSTEM QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    GIT_TAG ${NEOFOAM_KOKKOS_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Kokkos)
endif()

include(cmake/CPM.cmake)

cpmaddpackage(
  NAME
  cpptrace
  GITHUB_REPOSITORY
  jeremy-rifkin/cpptrace
  VERSION
  0.5.4
  SYSTEM)

cpmaddpackage(
  NAME
  nlohmann_json
  GITHUB_REPOSITORY
  nlohmann/json
  VERSION
  3.11.3
  SYSTEM)

cpmaddpackage(
  NAME
  spdlog
  GITHUB_REPOSITORY
  gabime/spdlog
  VERSION
  1.13.0
  SYSTEM)

cpmaddpackage(
  NAME
  cxxopts
  GITHUB_REPOSITORY
  jarro2783/cxxopts
  VERSION
  3.2.0
  SYSTEM)

if(NEOFOAM_ENABLE_GINKGO)
  set(GINKGO_BUILD_TESTS
      OFF
      CACHE INTERNAL "")
  set(GINKGO_BUILD_BENCHMARKS
      OFF
      CACHE INTERNAL "")
  set(GINKGO_BUILD_EXAMPLES
      OFF
      CACHE INTERNAL "")
  cpmaddpackage(
    NAME
    Ginkgo
    GITHUB_REPOSITORY
    ginkgo-project/ginkgo
    GIT_TAG
    batch-optim
    VERSION
    1.9.0
    SYSTEM)
endif()

if(NEOFOAM_ENABLE_PETSC)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc)
endif()

if(NEOFOAM_BUILD_TESTS OR NEOFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(
    NAME
    Catch2
    GITHUB_REPOSITORY
    catchorg/Catch2
    VERSION
    3.4.0
    SYSTEM)
endif()
