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
  set(oneValueKeywords "INCLUDE" "LIBNAME" "LIBPATH")
  set(multiValueKeywords "EXTRA_LINK_TARGET")
  cmake_parse_arguments("NF" "${options}" "${oneValueKeywords}" "${multiValueKeywords}" ${ARGN})
  if(NOT DEFINED "NF_LIBNAME")
    set(NF_LIBNAME ${NAME})
  endif()
  if(NOT DEFINED "NF_INCLUDE")
    set(NF_INCLUDE ${NAME})
  endif()
  if(NOT DEFINED "NF_LIBPATH")
    set(NF_LIBPATH $ENV{FOAM_LIBBIN})
  endif()

  set(OFINCDIR $ENV{FOAM_SRC}/${NF_INCLUDE}/lnInclude)
  set(OFLIBDIR ${NF_LIBPATH}/lib${NF_LIBNAME}${CMAKE_SHARED_LIBRARY_SUFFIX})

  add_library(OpenFOAM::${NAME} SHARED IMPORTED)
  set_target_properties(OpenFOAM::${NAME} PROPERTIES IMPORTED_LOCATION ${OFLIBDIR}
                                                     INTERFACE_INCLUDE_DIRECTORIES ${OFINCDIR})
  target_link_libraries(
    OpenFOAM
    PUBLIC
    INTERFACE OpenFOAM::${NAME} ${EXTRA_LINK_TARGET})
endfunction()

find_package(MPI REQUIRED)

add_library(OpenFOAM INTERFACE)

target_include_directories(OpenFOAM INTERFACE $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude
                                              $ENV{FOAM_SRC}/transportModels)
target_compile_definitions(OpenFOAM INTERFACE WM_LABEL_SIZE=$ENV{WM_LABEL_SIZE} NoRepository
                                              WM_$ENV{WM_PRECISION_OPTION} OPENFOAM=$ENV{FOAM_API})

importoflibrary(OpenFOAM)
importoflibrary(meshTools)
importoflibrary(finiteVolume)
importoflibrary(incompressibleTransportModels INCLUDE transportModels/incompressible)
importoflibrary(turbulenceModels INCLUDE TurbulenceModels/turbulenceModels)
importoflibrary(incompressibleTurbulenceModels INCLUDE TurbulenceModels/incompressible)
importoflibrary(
  Pstream
  INCLUDE
  Pstream/mpi
  LIBPATH
  $ENV{FOAM_LIBBIN}/$ENV{FOAM_MPI}
  EXTRA_LINK_TARGET
  MPI::MPI_CXX)
