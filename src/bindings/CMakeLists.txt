# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(
  Python 3.9
  COMPONENTS Interpreter ${DEV_MODULE}
  REQUIRED)

message(STATUS "Python_EXECUTABLE = ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS = ${Python_INCLUDE_DIRS}")

# Set output directory
if(DEFINED SKBUILD)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/neofoam)
endif()

nanobind_add_module(
  neofoam_bindings
  NB_SHARED
  NOMINSIZE
  neofoam.cpp
  runtime.cpp
  fieldFactories.cpp
  fieldWriters.cpp
  pdeSolver.cpp
  pressureVelocityCoupling.cpp
  utility.cpp
  explicitOperators.cpp
  turbulenceModel.cpp
  wallFunctions.cpp
  fieldRelaxation.cpp)

# Workaround for nvcc_wrapper failing on -Os which nanobind might add
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(neofoam_bindings PRIVATE -O3)
endif()

# Link against NeoFOAM
target_link_libraries(neofoam_bindings PRIVATE NeoFOAM)

# Installation
if(DEFINED SKBUILD)
  install(TARGETS neofoam_bindings LIBRARY DESTINATION neofoam COMPONENT python)
else()
  install(TARGETS neofoam_bindings LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neofoam)
endif()
