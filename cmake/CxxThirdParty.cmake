# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(cmake/CPM.cmake)
include(ExternalProject)

function(scorep_add_external_project)
    message(STATUS "[Score-P] Bootstrapping Score-P via ExternalProject_Add")

    # currently defaults to gcc need to be set to clang manually by the user
    set(SCOREP_COMPILER_SUITE "gcc" CACHE STRING "Score-P compiler suite (gcc, clang, intel, ...)")
    set_property(CACHE SCOREP_COMPILER_SUITE PROPERTY STRINGS gcc clang)

    message(STATUS "[Score-P] Compiler suite: ${SCOREP_COMPILER_SUITE}
                              Defaults to 'gcc', if this is not the intended 
			      compiler set manually via SCOREP_COMPILER_SUITE")

    ExternalProject_Add(scorep_build
        SOURCE_DIR     ${CMAKE_BINARY_DIR}/_deps/scorep-src
        URL            https://zenodo.org/records/17297069/files/scorep-9.3.tar.gz
	URL_HASH       MD5=661e52f439614e51164526d5a2c28b7a
        DOWNLOAD_NO_PROGRESS ON
	DOWNLOAD_EXTRACT_TIMESTAMP OFF
        CONFIGURE_COMMAND
            env CC=${CMAKE_C_COMPILER}
                CXX=${CMAKE_CXX_COMPILER}
            ${CMAKE_BINARY_DIR}/_deps/scorep-src/configure
                --prefix=${SCOREP_PREFIX}
                --with-libgotcha=download
                --with-libbfd=download
		--disable-cuda
                --with-nocross-compiler-suite=${SCOREP_COMPILER_SUITE}

        BUILD_COMMAND   make -j
        INSTALL_COMMAND make install
    )

    message(STATUS "[Score-P] Bootstrap scheduled → run: cmake --build --preset scorep 
            && and then reconfigure and build again")
endfunction()

if(NeoFOAM_BUILD_TESTS OR NeoFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(NAME Catch2 GITHUB_REPOSITORY catchorg/Catch2 VERSION 3.4.0)
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
