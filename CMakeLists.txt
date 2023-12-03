# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

cmake_minimum_required(VERSION 3.22.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "An implementation of FOAM")

include(cmake/PreventInSourceBuilds.cmake)
include(cmake/StandardProjectSettings.cmake)

option(NEOFOAM_DP_SCALAR "double precision scalar" TRUE)
option(NEOFOAM_DP_LABELS "double precision label" TRUE)
option(NEOFOAM_BUILD_APPS "Build the applications folder" TRUE)
option(NEOFOAM_BUILD_TESTS "Build the unit tests" TRUE)

option(NEOFOAM_ENABLE_SANITIZE_ADDRESS "Enable address sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_LEAK "Enable leak sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_UB "Enable undefined behaviour sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_THREAD "Enable thread sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_MEMORY "Enable memory sanitizer" FALSE)
option(NEOFOAM_ENABLE_CPP_CHECK "Enable cpp check static analyzer" FALSE)
option(NEOFOAM_ENABLE_CLANG_TIDY "Enable clang tidy static analyzer" FALSE)
option(NEOFOAM_ENABLE_PCH "Enable precompiled header" FALSE)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

add_library(project_options INTERFACE)
include(cmake/CompilerWarnings.cmake)
add_library(project_warnings INTERFACE)
myproject_set_project_warnings(project_warnings ${myproject_WARNINGS_AS_ERRORS}
                               FALSE "" "" "")

include(cmake/Sanitizer.cmake)
enable_sanitizers(
  project_options NEOFOAM_ENABLE_SANITIZE_ADDRESS NEOFOAM_ENABLE_SANITIZE_LEAK
  NEOFOAM_ENABLE_SANITIZE_UB NEOFOAM_ENABLE_SANITIZE_THREAD
  NEOFOAM_ENABLE_SANITIZE_MEMORY)

if(NEOFOAM_ENABLE_PCH)
  target_precompile_headers(project_options INTERFACE <vector> <string>
                            <utility>)
endif()

include(cmake/StaticAnalyzers.cmake)
if(NEOFOAM_ENABLE_CPP_CHECK)
  myproject_enable_cppcheck(FALSE "")
endif()
if(NEOFOAM_ENABLE_CLANG_TIDY)
  myproject_enable_cppcheck(FALSE "")
endif()

add_subdirectory(src)
if(NEOFOAM_BUILD_APPS)
  add_subdirectory(applications)
endif()

add_custom_target(
  cmake-format
  COMMAND find . -name CMakeLists.txt -exec cmake-format -i {} +
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

include(cmake/install_helpers.cmake)
neofoam_install()
