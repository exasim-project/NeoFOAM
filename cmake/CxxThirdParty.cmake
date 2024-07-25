# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(NEOFOAM_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NEOFOAM_KOKKOS_CHECKOUT_VERSION)

set(NEOFOAM_KOKKOS_KERNELS_CHECKOUT_VERSION
    ${NEOFOAM_KOKKOS_CHECKOUT_VERSION}
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NEOFOAM_KOKKOS_KERNELS_CHECKOUT_VERSION)

find_package(MPI 3.1 REQUIRED)
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

endif()

# set(KokkosKernels_ENABLE_PERFTESTS OFF CACHE BOOL "")
set(KokkosKernels_ENABLED_COMPONENTS
    "BATCHED;BLAS"
    CACHE STRING "" FORCE)
# set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "") set(KokkosKernels_ENABLE_BATCHED ON
# CACHE BOOL "") set(KokkosKernels_ENABLE_BLAS ON CACHE BOOL "") set(KokkosKernels_ENABLE_LAPACK OFF
# CACHE BOOL "") set(KokkosKernels_ENABLE_GRAPH OFF CACHE BOOL "") set(KokkosKernels_ENABLE_SPARSE
# OFF CACHE BOOL "") set(KokkosKernels_ENABLE_ODE OFF CACHE BOOL "")

FetchContent_Declare(
  KokkosKernels
  SYSTEM QUITE
  GIT_SHALLOW ON
  GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
  GIT_TAG ${NEOFOAM_KOKKOS_KERNELS_CHECKOUT_VERSION})

FetchContent_MakeAvailable(KokkosKernels)

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
  sundials
  GITHUB_REPOSITORY
  LLNL/sundials
  VERSION
  7.1.1
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
