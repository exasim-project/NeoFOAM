# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(cmake/CPM.cmake)

if(NeoFOAM_BUILD_TESTS OR NeoFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(NAME Catch2 GITHUB_REPOSITORY catchorg/Catch2 VERSION 3.4.0)
endif()

if(NEOFOAM_WITH_MPI)
  if(WIN32)
    message(FATAL_ERROR "NEOFOAM_WITH_MPI not supported on Windows")
  endif()
  find_package(MPI 3.1 REQUIRED)
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
    # grab the SHA from submodule automatically
    find_package(Git QUIET)
    if(EXISTS "${NeoFOAM_SOURCE_DIR}/.git" AND GIT_FOUND)
      # Note: it will not initialize the submodule and does not require initialization
      execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule status "src/NeoN"
        WORKING_DIRECTORY ${NeoFOAM_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_SUBMODULE_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      # the output format is -<SHA> src/NEON when the submodule uninitialization
      string(SUBSTRING "${GIT_SUBMODULE_STRING}" 0 1 SUBMODULE_PREFIX)
      string(SUBSTRING "${GIT_SUBMODULE_STRING}" 1 40 SUBMODULE_HASH)
      set(NEOFOAM_NEON_VERSION
          "${SUBMODULE_HASH}"
          CACHE INTERNAL "")
    else()
      # if is is not from git or user does not have git, it fallback to predefined value.
      set(NEOFOAM_NEON_VERSION
          "main"
          CACHE INTERNAL "")
    endif()
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
