# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(NEOFOAM_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NEOFOAM_KOKKOS_CHECKOUT_VERSION)
if(NEOFOAM_ENABLE_MPI_SUPPORT)
  if(WIN32)
    message(FATAL_ERROR "NEOFOAM_ENABLE_MPI_SUPPORT not supported on Windows")
  endif()
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

if(${NEOFOAM_WITH_SUNDIALS})

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
