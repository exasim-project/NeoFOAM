# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(cmake/CPM.cmake)

if(NeoFOAM_BUILD_TESTS OR NeoFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(NAME Catch2 GITHUB_REPOSITORY catchorg/Catch2 VERSION 3.4.0)
endif()

if(ENABLE_PROFILING)
cpmaddpackage(
    NAME ScoreP
    URL https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-9.3/scorep-9.3.tar.gz
    URL_HASH SHA256=5498b31b1d6c04b08a9d408320a7515e884538d248de58b6dd11b48c8f364112
    DOWNLOAD_ONLY YES
  )
  cpmaddpackage(
    NAME binutils
    URL https://ftp.gnu.org/gnu/binutils/binutils-2.43.tar.gz
    URL_HASH SHA256=025c436d15049076ebe511d29651cc4785ee502965a8839936a65518582bdd64
    DOWNLOAD_ONLY YES
  )
  include(ExternalProject)
  ExternalProject_Add(scorep_build
    SOURCE_DIR ${ScoreP_SOURCE_DIR}
    CONFIGURE_COMMAND ${ScoreP_SOURCE_DIR}/configure
        --prefix=${CMAKE_BINARY_DIR}/scorep
        --with-libgotcha=download
        --with-libbfd=download
        --with-cuda
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
  )
  ExternalProject_Get_Property(scorep_build INSTALL_DIR)
  set(SCOREP_ROOT ${INSTALL_DIR})
  set(ENV{SCOREP_ROOT} ${INSTALL_DIR})
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

