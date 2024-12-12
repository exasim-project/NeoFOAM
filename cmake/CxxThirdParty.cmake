# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(NEOFOAM_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NEOFOAM_KOKKOS_CHECKOUT_VERSION)
if(NEOFOAM_ENABLE_MPI_SUPPORT)
  find_package(MPI 3.1 REQUIRED)
endif()

find_package(Kokkos ${NEOFOAM_KOKKOS_CHECKOUT_VERSION} QUIET)

if(NOT ${Kokkos_FOUND})
  include(FetchContent)
  include(cmake/AutoEnableDevice.cmake)

  FetchContent_Declare(
    Kokkos
    SYSTEM QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    GIT_TAG ${NEOFOAM_KOKKOS_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Kokkos)
else()
  message(STATUS "Found Kokkos ${NEOFOAM_KOKKOS_CHECKOUT_VERSION}")
endif()

# set(KokkosKernels_ENABLE_PERFTESTS OFF CACHE BOOL "") set(KokkosKernels_ENABLED_COMPONENTS
# "BATCHED;BLAS" CACHE STRING "" FORCE) set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
# set(KokkosKernels_ENABLE_BATCHED ON CACHE BOOL "") set(KokkosKernels_ENABLE_BLAS ON CACHE BOOL "")
# set(KokkosKernels_ENABLE_LAPACK OFF CACHE BOOL "") set(KokkosKernels_ENABLE_GRAPH OFF CACHE BOOL
# "") set(KokkosKernels_ENABLE_SPARSE OFF CACHE BOOL "") set(KokkosKernels_ENABLE_ODE OFF CACHE BOOL
# "")

# FetchContent_Declare( KokkosKernels SYSTEM QUITE GIT_SHALLOW ON GIT_REPOSITORY
# "https://github.com/kokkos/kokkos-kernels.git" GIT_TAG ${NEOFOAM_KOKKOS_KERNELS_CHECKOUT_VERSION})

# FetchContent_MakeAvailable(KokkosKernels)

include(cmake/CPM.cmake)

cpmaddpackage(
  NAME
  cpptrace
  URL
  https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v0.7.3.zip
  VERSION
  0.7.3
  SYSTEM)

cpmaddpackage(
  NAME
  nlohmann_json
  URL
  https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
  VERSION
  3.11.3
  SYSTEM)

cpmaddpackage(
  NAME
  sundials
  GITHUB_REPOSITORY
  LLNL/sundials
  VERSION
  7.1.1
  SYSTEM)

cpmaddpackage(
  NAME
  spdlog
  URL
  https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.zip
  VERSION
  1.13.0
  SYSTEM)

cpmaddpackage(
  NAME
  cxxopts
  URL
  https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
  VERSION
  3.2.0
  SYSTEM)

if(NEOFOAM_BUILD_TESTS OR NEOFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(
    NAME
    Catch2
    URL
    https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.zip
    VERSION
    3.4.0
    SYSTEM)
endif()
