# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

cmake_minimum_required(VERSION 3.22.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "An implementation of FOAM")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

option(FORCE_NEOFOAM_BUILD_KOKKOS
       "Forces NeoFOAM to build kokkos even if already present" FALSE)

# Add the cmake folder so the find_package command finds custom packages
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(NEOFOAM_DP_SCALAR "double precision scalar" TRUE)
option(NEOFOAM_DP_LABELS "double precision label" TRUE)
option(NEOFOAM_BUILD_WITH_APPS "Build the applications folder" TRUE)
option(NEOFOAM_BUILD_WITH_TESTS "Build the unit tests" TRUE)
option(NEOFOAM_BUILD_APPS "Build the applications folder" TRUE)
option(NEOFOAM_BUILD_TESTS "Build the unit tests" TRUE)
option(NEOFOAM_BUILD_BENCHMARKS "Build benchmarks" TRUE)
option(NEOFOAM_BUILD_DOC "Build documentation" FALSE)

option(NEOFOAM_ENABLE_SANITIZE_ADDRESS "Enable address sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_LEAK "Enable leak sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_UB "Enable undefined behaviour sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_THREAD "Enable thread sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_MEMORY "Enable memory sanitizer" FALSE)
option(NEOFOAM_ENABLE_CPP_CHECK "Enable cpp check static analyzer" FALSE)
option(NEOFOAM_ENABLE_CLANG_TIDY "Enable clang tidy static analyzer" FALSE)
option(NEOFOAM_ENABLE_PCH "Enable precompiled header" FALSE)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

include(cmake/Kokkos.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/StandardProjectSettings.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizer.cmake)
include(cmake/StaticAnalyzers.cmake)

add_library(project_options INTERFACE)
add_library(project_warnings INTERFACE)
myproject_set_project_warnings(project_warnings ${myproject_WARNINGS_AS_ERRORS}
                               FALSE "" "" "")

if(NEOFOAM_BUILD_PCH)
  target_precompile_headers(project_options INTERFACE <vector> <string>
                            <utility>)
endif()

if(NEOFOAM_ENABLE_CPP_CHECK)
  neofoam_enable_cppcheck(FALSE "")
endif()
if(NEOFOAM_ENABLE_CLANG_TIDY)
  neofoam_enable_clang_tidy(FALSE "")
endif()

add_subdirectory(src)
if(NEOFOAM_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(NEOFOAM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(${NEOFOAM_DP_SCALAR})
  target_compile_definitions(NeoFOAM PUBLIC NEOFOAM_DP_SCALAR=1)
endif()
if(${NEOFOAM_DP_LABEL})
  target_compile_definitions(NeoFOAM PUBLIC NEOFOAM_DP_LABEL=1)
endif()

if(NEOFOAM_BUILD_WITH_APPS)
  add_subdirectory(applications)
endif()

if(NEOFOAM_BUILD_DOC)
  include(cmake/Docs.cmake)
  neofoam_build_docs()
endif()

add_custom_target(
  cmake-format
  COMMAND find . -name CMakeLists.txt -exec cmake-format -i {} +
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_custom_target(
  format
  COMMAND find  include src applications test benchmarks -name "*.[c,h]*pp" -exec
          clang-format -i {} +
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

include(cmake/InstallHelpers.cmake)
neofoam_install()
