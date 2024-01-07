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

include(cmake/PreventInSourceBuilds.cmake)
include(cmake/StandardProjectSettings.cmake)

option(BUILD_NEOFOAM_DP_SCALAR "double precision scalar" TRUE)
option(BUILD_NEOFOAM_DP_LABELS "double precision label" TRUE)
option(BUILD_NEOFOAM_WITH_APPS "Build the applications folder" TRUE)
option(BUILD_NEOFOAM_WITH_TESTS "Build the unit tests" TRUE)
option(FORCE_NEOFOAM_BUILD_KOKKOS
       "Forces NeoFOAM to build kokkos even if already present" FALSE)

option(ENABLE_NEOFOAM_SANITIZE_ADDRESS "Enable address sanitizer" FALSE)
option(ENABLE_NEOFOAM_SANITIZE_LEAK "Enable leak sanitizer" FALSE)
option(ENABLE_NEOFOAM_SANITIZE_UB "Enable undefined behaviour sanitizer" FALSE)
option(ENABLE_NEOFOAM_SANITIZE_THREAD "Enable thread sanitizer" FALSE)
option(ENABLE_NEOFOAM_SANITIZE_MEMORY "Enable memory sanitizer" FALSE)
option(ENABLE_NEOFOAM_CPP_CHECK "Enable cpp check static analyzer" FALSE)
option(ENABLE_NEOFOAM_CLANG_TIDY "Enable clang tidy static analyzer" FALSE)
option(ENABLE_NEOFOAM_PCH "Enable precompiled header" FALSE)

include(cmake/kokkos.cmake)

add_library(project_options INTERFACE)
include(cmake/CompilerWarnings.cmake)
add_library(project_warnings INTERFACE)
myproject_set_project_warnings(project_warnings ${myproject_WARNINGS_AS_ERRORS}
                               FALSE "" "" "")

if(BUILD_NEOFOAM_PCH)
  target_precompile_headers(project_options INTERFACE <vector> <string>
                            <utility>)
endif()

include(cmake/Sanitizer.cmake)

include(cmake/StaticAnalyzers.cmake)
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
