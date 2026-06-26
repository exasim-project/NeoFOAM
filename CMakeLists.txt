# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

cmake_minimum_required(VERSION 3.22.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "An implementation of FOAM")

if(NOT DEFINED ENV{FOAM_SRC})
  message(FATAL_ERROR "You must source OpenFOAM before building NeoFOAM")
endif()
if(NOT $ENV{FOAM_API} GREATER_EQUAL 2406)
  message(WARNING "OpenFOAM version < 2406")
endif()

option(NEOFOAM_BUILD_EXAMPLES "Build the NeoN examples" ON)
option(NEOFOAM_BUILD_TESTS "Build the unit tests" OFF)
option(NEOFOAM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NEOFOAM_WITH_MPI "Enable MPI support" ON)
option(NEOFOAM_ENABLE_KOKKOS_TOOLS "Enable Kokkos Tools" OFF)

option(NEOFOAM_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# Detect if building with scikit-build-core
if(DEFINED SKBUILD)
  message(STATUS "Building with scikit-build-core")
  set(NEOFOAM_BUILD_PYTHON_BINDINGS
      ON
      CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS
      ON
      CACHE BOOL "" FORCE)

  # Install layout (wheel build): route every CMake-installed file under the neofoam/ package dir.
  # Set before NeoN and its deps are added so they inherit it (via GNUInstallDirs). This keeps the
  # bundled deps' *Config.cmake out of the site-packages prefix root, where NeoN's
  # find_package(Kokkos/Ginkgo/nlohmann_json) would otherwise pick them up and flip the include
  # paths build-tree<->installed on every `pip install` (recompiling all of NeoN). The Python
  # bindings install to explicit destinations (neon/, neofoam/), so they stay importable.
  set(CMAKE_INSTALL_INCLUDEDIR "neofoam/include")
  set(CMAKE_INSTALL_LIBDIR "neofoam/lib")
  set(CMAKE_INSTALL_BINDIR "neofoam/bin")
  set(CMAKE_INSTALL_DATAROOTDIR "neofoam/share")

  # All native libs land in neofoam/lib. $ORIGIN(/lib) covers libs/bindings inside neofoam/;
  # $ORIGIN/../lib and $ORIGIN/../../lib reach a couple of deps (camp, cpptrace) that hardcode lib/
  # and leak to the site-packages root.
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN" "$ORIGIN/lib" "$ORIGIN/../lib" "$ORIGIN/../../lib")
endif()

set(NEOFOAM_BENCHMARK_MODE
    "all"
    CACHE STRING "perform only a subset of the benchmarks")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Add the cmake folder so the find_package command finds custom packages
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set the output directories for all binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(cmake/OpenFOAM.cmake)
include(cmake/KokkosTools.cmake)

set(NeoN_WITH_MPI
    ${NEOFOAM_WITH_MPI}
    CACHE BOOL "" FORCE)
if(NOT DEFINED NEOFOAM_NEON_DIR)
  if(EXISTS ${PROJECT_SOURCE_DIR}/src/NeoN/CMakeLists.txt AND NOT DEFINED NEOFOAM_NEON_VERSION)
    message(STATUS "src/NeoN/CMakeLists.txt found, using NeoN")
    add_subdirectory(src/NeoN)
  else()
    message(STATUS "src/NeoN/CMakeLists.txt not found, using CPM")
    set(NEOFOAM_NEON_VIA_CPM ON)
  endif()
else()
  message(STATUS "using NEOFOAM_NEON_DIR ${NEOFOAM_NEON_DIR}")
  add_subdirectory(${NEOFOAM_NEON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/NeoN)
endif()

include(cmake/CxxThirdParty.cmake)

# NeoN hard-sets the _neon binding's INSTALL_RPATH to "$ORIGIN/../lib" (correct only for the old
# flat layout). _neon installs to site-packages/neon/ while the native libs live in neofoam/lib, so
# point it at $ORIGIN/../neofoam/lib (+ $ORIGIN/../lib for camp/cpptrace, which leak to the root).
if(DEFINED SKBUILD AND TARGET _neon)
  set_target_properties(_neon PROPERTIES INSTALL_RPATH "$ORIGIN/../neofoam/lib;$ORIGIN/../lib")
endif()

# Umpire/camp are CPM deps added inside src/NeoN, where NeoN sets CMAKE_INSTALL_RPATH to the
# absolute install prefix /lib (the wheel-staging temp dir under SKBUILD). That path is gone after
# install, so libumpire.so cannot find its direct dependency libcamp.so at import (camp/cpptrace
# "leak" to the site-packages root lib/). Re-point them at the installed layout with $ORIGIN-
# relative entries: from neofoam/lib, $ORIGIN reaches its siblings and $ORIGIN/../../lib reaches the
# site-packages root lib/ where camp lives. This only rewrites the install RPATH (a relink), so it
# does not trigger a recompile.
if(DEFINED SKBUILD)
  foreach(_leaked_dep umpire camp)
    if(TARGET ${_leaked_dep})
      set_target_properties(
        ${_leaked_dep}
        PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;$ORIGIN/../lib;$ORIGIN/../../lib"
                   INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()
  endforeach()
endif()

add_subdirectory(include)
add_subdirectory(src)

if(NEOFOAM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(NEOFOAM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(NEOFOAM_BUILD_BENCHMARKS)
  enable_testing()
  add_subdirectory(benchmarks)
endif()

if(NEOFOAM_BUILD_PYTHON_BINDINGS)
  add_subdirectory(src/bindings)
endif()

include(cmake/banner.cmake)
