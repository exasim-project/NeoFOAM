# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoN authors

set(NeoN_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NeoN_KOKKOS_CHECKOUT_VERSION)
if(NeoN_ENABLE_MPI_SUPPORT)
  if(WIN32)
    message(FATAL_ERROR "NeoN_ENABLE_MPI_SUPPORT not supported on Windows")
  endif()
  find_package(MPI 3.1 REQUIRED)
endif()

find_package(Kokkos ${NeoN_KOKKOS_CHECKOUT_VERSION} QUIET)

if(NOT ${Kokkos_FOUND})
  include(FetchContent)
  include(cmake/AutoEnableDevice.cmake)

  FetchContent_Declare(
    Kokkos
    SYSTEM QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    GIT_TAG ${NeoN_KOKKOS_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Kokkos)
else()
  message(STATUS "Found Kokkos ${NeoN_KOKKOS_CHECKOUT_VERSION}")
endif()

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
  VERSION
  3.11.3
  URL
  https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
  SYSTEM)

if(${NeoN_WITH_ADIOS2})

  set(ADIOS2_KOKKOS_PATCH git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/adios2_kokkos.patch)

  set(ADIOS2_OPTIONS
      "BUILD_TYPE ${CMAKE_BUILD_TYPE}"
      "ADIOS2_USE_Kokkos ON"
      "Kokkos_DIR ${Kokkos_BINARY_DIR}"
      "ADIOS2_USE_Fortran OFF"
      "ADIOS2_USE_Python OFF"
      "ADIOS2_USE_MHS OFF"
      "ADIOS2_USE_SST OFF"
      "ADIOS2_BUILD_EXAMPLES OFF"
      "BUILD_TESTING OFF"
      "ADIOS2_USE_Profiling OFF")

  if(WIN32)
    list(APPEND ADIOS2_OPTIONS "BUILD_STATIC_LIBS ON")
    list(APPEND ADIOS2_OPTIONS "BUILD_SHARED_LIBS OFF")
  else()
    list(APPEND ADIOS2_OPTIONS "BUILD_STATIC_LIBS OFF")
    list(APPEND ADIOS2_OPTIONS "BUILD_SHARED_LIBS ON")
  endif()

  cpmaddpackage(
    NAME
    adios2
    GITHUB_REPOSITORY
    ornladios/ADIOS2
    PATCH_COMMAND
    ${ADIOS2_KOKKOS_PATCH}
    VERSION
    2.10.2
    OPTIONS
    ${ADIOS2_OPTIONS}
    ${ADIOS2_CUDA_OPTIONS}
    SYSTEM)
endif()

if(${NeoN_WITH_SUNDIALS})

  set(SUNDIALS_OPTIONS
      "BUILD_TESTING OFF"
      "EXAMPLES_INSTALL OFF"
      "BUILD_ARKODE ON"
      "BUILD_CVODE OFF"
      "BUILD_CVODES OFF"
      "BUILD_IDA OFF"
      "BUILD_IDAS OFF"
      "BUILD_KINSOL OFF"
      "BUILD_CPODES OFF")

  if(WIN32)
    list(APPEND SUNDIALS_OPTIONS "BUILD_STATIC_LIBS ON")
    list(APPEND SUNDIALS_OPTIONS "BUILD_SHARED_LIBS OFF")
  else()
    list(APPEND SUNDIALS_OPTIONS "BUILD_STATIC_LIBS OFF")
    list(APPEND SUNDIALS_OPTIONS "BUILD_SHARED_LIBS ON")
  endif()

  if(Kokkos_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(SUNDIALS_CUDA_OPTIONS "ENABLE_CUDA ON" "SUNDIALS_BUILD_KOKKOS ON")
  else()
    set(SUNDIALS_CUDA_OPTIONS "ENABLE_CUDA OFF" "SUNDIALS_BUILD_KOKKOS ON")
  endif()

  cpmaddpackage(
    NAME
    sundials
    GITHUB_REPOSITORY
    LLNL/sundials
    VERSION
    7.1.1
    OPTIONS
    ${SUNDIALS_OPTIONS}
    ${SUNDIALS_CUDA_OPTIONS}
    SYSTEM)
endif()

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

if(${NeoN_WITH_GINKGO})
  cpmaddpackage(
    NAME
    Ginkgo
    VERSION
    1.10.0
    GITHUB_REPOSITORY
    ginkgo-project/ginkgo
    GIT_TAG
    kokkos-threads
    OPTIONS
    "GINKGO_BUILD_TESTS OFF"
    "GINKGO_BUILD_BENCHMARKS OFF"
    "GINKGO_BUILD_EXAMPLES OFF"
    "GINKGO_ENABLE_HALF OFF"
    "GINKGO_BUILD_MPI ${NeoN_ENABLE_MPI_SUPPORT}"
    "GINKGO_BUILD_CUDA ${Kokkos_ENABLE_CUDA}"
    "GINKGO_BUILD_HIP ${Kokkos_ENABLE_HIP}"
    SYSTEM)
endif()

if(NeoN_BUILD_TESTS OR NeoN_BUILD_BENCHMARKS)
  cpmaddpackage(
    NAME
    Catch2
    URL
    https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.zip
    VERSION
    3.4.0
    SYSTEM)
endif()
