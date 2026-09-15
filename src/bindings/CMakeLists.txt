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
  fieldRelaxation.cpp
  mrf.cpp)

# nanobind builds the module -fvisibility=hidden, which also hides the typeinfo emitted here for the
# NeoN types this module boxes (VolumeField and friends). std::any dispatches on type_info identity,
# so a field boxed in this module is unrecognisable to an any_cast compiled into libNeoN --
# rotate_old_times() then throws "bad any cast" with requested and actual printing the same name.
# NeoN marks its own Dictionary/TokenList NEON_TYPE_VISIBLE for this reason (see
# NeoN/core/visibility.hpp); default visibility does it for every type crossing this boundary.
# nanobind's own symbols stay hidden: NB_NAMESPACE carries visibility("hidden") itself.
set_target_properties(neofoam_bindings PROPERTIES CXX_VISIBILITY_PRESET default)

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
