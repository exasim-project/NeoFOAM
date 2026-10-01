# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# cmake-format: off
# This function imports an OpenFOAM library
# It requires:
#  * NAME the library name and assumes a lib${Name}.so exists
# following optional keywords:
# * INCLUDE the library lnInclude path
# * LIBNAME alternative library name otherwise lib{Name}.so is taken
# * LIBPATH alternative path to search for .so file
# cmake-format: on
function(importOFLibrary NAME)
  set(options "")
  set(oneValueKeywords "LIBNAME" "LIBPATH")
  set(multiValueKeywords "INCLUDE_LN" "INCLUDE_ROOT" "EXTRA_LINK_TARGET")
  cmake_parse_arguments("NF" "${options}" "${oneValueKeywords}" "${multiValueKeywords}" ${ARGN})

  if(NOT DEFINED NF_LIBNAME)
    set(NF_LIBNAME ${NAME})
  endif()
  if(NOT DEFINED NF_LIBPATH)
    set(NF_LIBPATH $ENV{FOAM_LIBBIN})
  endif()

  # Default: OpenFOAM libs typically expose headers via <module>/lnInclude
  if(NOT DEFINED NF_INCLUDE_LN AND NOT DEFINED NF_INCLUDE_ROOT)
    set(NF_INCLUDE_LN ${NAME})
  endif()

  set(OFINCDIRS "")
  foreach(inc IN LISTS NF_INCLUDE_LN)
    list(APPEND OFINCDIRS "$ENV{FOAM_SRC}/${inc}/lnInclude")
  endforeach()
  foreach(inc IN LISTS NF_INCLUDE_ROOT)
    list(APPEND OFINCDIRS "$ENV{FOAM_SRC}/${inc}")
  endforeach()

  set(OFLIBDIR ${NF_LIBPATH}/lib${NF_LIBNAME}${CMAKE_SHARED_LIBRARY_SUFFIX})

  add_library(OpenFOAM::${NAME} SHARED IMPORTED)
  set_target_properties(OpenFOAM::${NAME} PROPERTIES IMPORTED_LOCATION ${OFLIBDIR}
                                                     INTERFACE_INCLUDE_DIRECTORIES "${OFINCDIRS}")
  target_link_libraries(
    OpenFOAM
    PUBLIC
    INTERFACE OpenFOAM::${NAME} ${EXTRA_LINK_TARGET})
endfunction()

# find_package(MPI REQUIRED)

add_library(OpenFOAM INTERFACE)
target_include_directories(
  OpenFOAM
  PUBLIC
  INTERFACE $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude)
target_compile_definitions(OpenFOAM INTERFACE WM_LABEL_SIZE=$ENV{WM_LABEL_SIZE} NoRepository
                                              WM_$ENV{WM_PRECISION_OPTION} OPENFOAM=$ENV{FOAM_API})

# OpenFOAM exposes its label width via WM_LABEL_SIZE (32/64). Configure NeoN's corresponding CMake
# option before it is added as a subdirectory so that NeoN can set its own public compile
# definitions (NeoN_DP_LABEL) consistently.
if($ENV{WM_LABEL_SIZE} EQUAL 64)
  set(NeoN_DEFINE_DP_LABEL
      ON
      CACHE BOOL "Use 64-bit labels to match OpenFOAM" FORCE)
else()
  set(NeoN_DEFINE_DP_LABEL
      OFF
      CACHE BOOL "Use 32-bit labels to match OpenFOAM" FORCE)
endif()

importoflibrary(OpenFOAM)
importoflibrary(meshTools)
importoflibrary(finiteVolume)
importoflibrary(incompressibleTransportModels INCLUDE_ROOT transportModels INCLUDE_LN
                transportModels/incompressible)
importoflibrary(turbulenceModels INCLUDE_LN TurbulenceModels/turbulenceModels)
importoflibrary(incompressibleTurbulenceModels INCLUDE_LN TurbulenceModels/incompressible)
importoflibrary(
  Pstream
  INCLUDE_LN
  Pstream/mpi
  LIBPATH
  $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}
  EXTRA_LINK_TARGET
  MPI::MPI_CXX)
importoflibrary(forces INCLUDE_LN functionObjects/forces)

# pybFoam compatibility.
#
# pybFoam is built in-tree through CPM (see CMakeLists.txt), and its cmake/FindOpenFOAM.cmake
# creates its OpenFOAM::* targets guarded by `if(NOT TARGET ...)`. Ours are defined first, so for
# the two names that overlap — meshTools and finiteVolume — pybFoam silently inherits ours instead
# of creating its own. Its sources then fail to compile with "messageStream.H: No such file or
# directory", because our targets carry only their own lnInclude while pybFoam's pull in the core
# headers transitively.
#
# Give those two targets what pybFoam's equivalents provide. Purely additive: these paths and
# libraries are correct for the modules in question regardless of pybFoam, and nothing NeoFOAM
# already builds changes.
#
# OpenFOAM::OpenFOAM also gains OSspecific, which pybFoam's OpenFOAM::core carries and which NeoFOAM
# previously had only on the aggregate `OpenFOAM` interface target.
if(TARGET OpenFOAM::OpenFOAM)
  target_include_directories(OpenFOAM::OpenFOAM INTERFACE $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude)
endif()

if(TARGET OpenFOAM::meshTools)
  target_include_directories(OpenFOAM::meshTools INTERFACE $ENV{FOAM_SRC}/dynamicMesh/lnInclude)
  target_link_libraries(OpenFOAM::meshTools INTERFACE OpenFOAM::OpenFOAM)
endif()

if(TARGET OpenFOAM::finiteVolume)
  target_include_directories(OpenFOAM::finiteVolume INTERFACE $ENV{FOAM_SRC}/dynamicFvMesh/lnInclude
                                                              $ENV{FOAM_SRC}/dynamicMesh/lnInclude)
  target_link_libraries(OpenFOAM::finiteVolume INTERFACE OpenFOAM::meshTools OpenFOAM::OpenFOAM)
endif()
