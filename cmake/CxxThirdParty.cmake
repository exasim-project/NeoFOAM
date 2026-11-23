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

# Only the private neofoam._vtkhdf module links HDF5. EXCLUDE_FROM_ALL also drops HDF5's install
# rules, so nothing of it lands in the wheel besides what is linked into that module.
if(NEOFOAM_WITH_HDF5 AND NEOFOAM_BUILD_PYTHON_BINDINGS)
  cpmaddpackage(
    NAME
    HDF5
    GITHUB_REPOSITORY
    HDFGroup/hdf5
    GIT_TAG
    hdf5_1.14.6
    GIT_SHALLOW
    TRUE
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    OPTIONS
    "HDF5_ENABLE_PARALLEL ${NEOFOAM_WITH_MPI}"
    "BUILD_SHARED_LIBS OFF"
    "BUILD_STATIC_LIBS ON"
    "CMAKE_POSITION_INDEPENDENT_CODE ON"
    "BUILD_TESTING OFF"
    "HDF5_BUILD_TOOLS OFF"
    "HDF5_BUILD_EXAMPLES OFF"
    "HDF5_BUILD_HL_LIB OFF"
    "HDF5_BUILD_CPP_LIB OFF"
    "HDF5_BUILD_FORTRAN OFF"
    "HDF5_ENABLE_Z_LIB_SUPPORT OFF"
    "HDF5_ENABLE_SZIP_SUPPORT OFF")
endif()

# nanobind is not fetched here: NeoN's CxxThirdParty.cmake provides it whenever
# NeoN_BUILD_PYTHON_BINDINGS is ON, taking it from the active Python environment
# (NeoN_EXTERNAL_NANOBIND, set by the python-bindings preset) so this module, _neon and pybFoam are
# built with one nanobind ABI and can exchange types.

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
