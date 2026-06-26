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

  # Configure RPATH for Python extension modules. Native artifacts are installed per-package (NeoN +
  # its bundled deps under neon/, NeoFOAM under neofoam/, see install-dir routing below), so a
  # binding/lib may need to resolve siblings in either package's lib dir. Cover all the relative
  # hops with $ORIGIN entries.
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_INSTALL_RPATH
      "$ORIGIN"
      "$ORIGIN/lib"
      "$ORIGIN/../lib"
      "$ORIGIN/../neon/lib"
      "$ORIGIN/../../neon/lib"
      "$ORIGIN/../neofoam/lib"
      "$ORIGIN/../../neofoam/lib")
endif()

# Per-package install layout (wheel build): route every CMake-installed file into its owning Python
# package so each ships its own libs + dev files (headers + CMake configs) and is independently
# reusable — and, critically, so the bundled deps' *Config.cmake do NOT land at the site-packages
# prefix root where NeoN's find_package() would pick them up and flip the include paths
# build-tree<->installed on every `pip install` (recompiling all of NeoN). NeoN and its deps
# (Kokkos/Ginkgo/Umpire/...) -> neon/ ; set before they are added.
if(DEFINED SKBUILD)
  set(CMAKE_INSTALL_INCLUDEDIR "neon/include")
  set(CMAKE_INSTALL_LIBDIR "neon/lib")
  set(CMAKE_INSTALL_BINDIR "neon/bin")
  set(CMAKE_INSTALL_DATAROOTDIR "neon/share")
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

if(NEOFOAM_USE_SYSTEM_NEON)
  find_package(NeoN CONFIG REQUIRED)
else()
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
endif()

include(cmake/CxxThirdParty.cmake)

# NeoN + its deps are now added; route the remaining NeoFOAM-owned artifacts under neofoam/.
if(DEFINED SKBUILD)
  set(CMAKE_INSTALL_INCLUDEDIR "neofoam/include")
  set(CMAKE_INSTALL_LIBDIR "neofoam/lib")
  set(CMAKE_INSTALL_BINDIR "neofoam/bin")
  set(CMAKE_INSTALL_DATAROOTDIR "neofoam/share")

  # NeoN hard-sets the _neon binding's INSTALL_RPATH to "$ORIGIN/../lib" (correct only for the old
  # flat layout). Now _neon lives at site-packages/neon/_neon.so and its libs at neon/lib, so point
  # it at $ORIGIN/lib. $ORIGIN/lib -> neon/lib (NeoN + most deps); $ORIGIN/../lib ->
  # site-packages/lib, a fallback for deps (camp, cpptrace) whose own install rules hardcode lib/
  # and ignore CMAKE_INSTALL_LIBDIR.
  if(TARGET _neon)
    set_target_properties(_neon PROPERTIES INSTALL_RPATH "$ORIGIN/lib;$ORIGIN;$ORIGIN/../lib")
  endif()
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
