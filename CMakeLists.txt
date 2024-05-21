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

# Add the cmake folder so the find_package command finds custom packages
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(NEOFOAM_DP_SCALAR "double precision scalar" TRUE)
option(NEOFOAM_DP_LABELS "double precision label" FALSE)
option(NEOFOAM_BUILD_APPS "Build the applications folder" TRUE)
option(NEOFOAM_DEVEL_TOOLS "Add development tools to the build system" FALSE)
option(NEOFOAM_BUILD_TESTS "Build the unit tests" FALSE)
option(NEOFOAM_BUILD_BENCHMARKS "Build benchmarks" FALSE)
option(NEOFOAM_BUILD_DOC "Build documentation" FALSE)

option(NEOFOAM_ENABLE_SANITIZE_ADDRESS "Enable address sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_LEAK "Enable leak sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_UB "Enable undefined behaviour sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_THREAD "Enable thread sanitizer" FALSE)
option(NEOFOAM_ENABLE_SANITIZE_MEMORY "Enable memory sanitizer" FALSE)
option(NEOFOAM_ENABLE_CPP_CHECK "Enable cpp check static analyzer" FALSE)
option(NEOFOAM_ENABLE_CLANG_TIDY "Enable clang tidy static analyzer" FALSE)
option(NEOFOAM_ENABLE_PCH "Enable precompiled header" FALSE)
option(NEOFOAM_ENABLE_IWYU "Enable iwyu checks" FALSE)

if(NOT DEFINED CPM_USE_LOCAL_PACKAGES)
  message(STATUS "Set CPM_USE_LOCAL_PACKAGES=ON by default.")
  set(CPM_USE_LOCAL_PACKAGES ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
  message(STATUS "Enabling generation of compilation database.")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

include(cmake/CxxThirdParty.cmake)
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
  enable_testing()
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

if(NEOFOAM_BUILD_APPS)
  add_subdirectory(applications)
endif()

if(NEOFOAM_BUILD_DOC)
  include(cmake/Docs.cmake)
  neofoam_build_docs()
endif()

if(NEOFOAM_DEVEL_TOOLS)
  find_program(PRE_COMMIT pre-commit)
  if(NOT PRE_COMMIT)
    message(
      FATAL_ERROR
        "The pre-commit command was not found. "
        "It is necessary if you want to commit changes to NeoFOAM. "
        "If that is not the case, set NEOFOAM_DEVEL_TOOLS=OFF. "
        "Otherwise install pre-commit via pipx (or pip) using:\n"
        "    pipx install pre-commit")
  endif()

  execute_process(
    COMMAND "${PRE_COMMIT}" "install"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE pre-commit-result
    OUTPUT_VARIABLE pre-commit-output
    ERROR_VARIABLE pre-commit-error)
  if(pre-commit-result)
    message(
      FATAL_ERROR
        "Failed to install the git hooks via pre-commit."
        "Please check the error message:\n"
        "${pre-commit-output}\n${pre-commit-error}")
  endif()
  if(pre-commit-output MATCHES "^Running in migration mode with existing hooks")
    message(
      WARNING
        "An existing git hook was encountered during `pre-commit install`."
        "The old git hook will also be executed."
        "Consider removing it with `pre-commit install -f`")
  elseif(NOT pre-commit-output MATCHES "^pre-commit installed at")
    message(WARNING "`pre-commit install` did not exit normally."
                    "Please check the output message:\n" "${pre-commit-output}")
  endif()

  add_custom_target(
    pre-commit
    COMMENT "execute precommit"
    COMMAND bash -c "${PRE_COMMIT} run"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM)
endif()

if(NEOFOAM_ENABLE_IWYU)
  find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
  set(IWYU_PATH_AND_OPTIONS ${iwyu_path})
  set_property(TARGET NeoFOAM PROPERTY CXX_INCLUDE_WHAT_YOU_USE
                                       ${IWYU_PATH_AND_OPTIONS})
endif()

# include(cmake/InstallHelpers.cmake) neofoam_install()
