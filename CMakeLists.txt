# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
cmake_minimum_required(VERSION 3.22.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "An implementation of FOAM")

# include kokkos.cmake before setting the project standard
# to avoid issues with nvcc not supporting c++20 standard

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

option(NEOFOAM_DP_SCALAR "double precision scalar" TRUE)
option(NEOFOAM_DP_LABELS "double precision label" TRUE)
option(NEOFOAM_WITH_APPS "Build the applications folder" TRUE)
option(NEOFOAM_WITH_TESTS "Build the unit tests" TRUE)
option(NEOFOAM_CUDA "Build with CUDA enabled " FALSE)
option(NEOFOAM_OMP "Build with OpenMP enabled" FALSE)

option(NEOFOAM_SANITIZE_ADDRESS "Enable address sanitizer" FALSE)
option(NEOFOAM_SANITIZE_LEAK "Enable leak sanitizer" FALSE)
option(NEOFOAM_SANITIZE_UB "Enable undefined behaviour sanitizer" FALSE)
option(NEOFOAM_SANITIZE_THREAD "Enable thread sanitizer" FALSE)
option(NEOFOAM_SANITIZE_MEMORY "Enable memory sanitizer" FALSE)
option(NEOFOAM_CPP_CHECK "Enable cpp check static analyzer" FALSE)
option(NEOFOAM_CLANG_TIDY "Enable clang tidy static analyzer" FALSE)
option(NEOFOAM_PCH "Enable precompiled header" FALSE)


include(cmake/kokkos.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/StandardProjectSettings.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizer.cmake)
include(cmake/StaticAnalyzers.cmake)

add_library(project_options INTERFACE)
add_library(project_warnings INTERFACE)
myproject_set_project_warnings(project_warnings ${myproject_WARNINGS_AS_ERRORS}
                               FALSE "" "" "")

if(BUILD_NEOFOAM_PCH)
  target_precompile_headers(project_options INTERFACE <vector> <string>
                            <utility>)
endif()

if(ENABLE_NEOFOAM_CPP_CHECK)
  neofoam_enable_cppcheck(FALSE "")
endif()
if(ENABLE_NEOFOAM_CLANG_TIDY)
  neofoam_enable_clang_tidy(FALSE "")
endif()

add_subdirectory(src)
if(NEOFOAM_BUILD_APPS)
  add_subdirectory(applications)
endif()

add_custom_target(
  cmake-format
  COMMAND find . -name CMakeLists.txt -exec cmake-format -i {} +
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_custom_target(
  format
  COMMAND find src applications -name "*.[c,h]*pp" -exec clang-format -i {} +
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

include(cmake/install_helpers.cmake)
neofoam_install()
