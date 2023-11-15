# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

cmake_minimum_required(VERSION 3.26.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  DESCRIPTION "An implementation of FOAM")

option(NeoFOAM "double precision scalar" TRUE)
option(NeoFOAM "double precision label" TRUE)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include_directories(${googletest_SOURCE_DIR}/googletest/include)
enable_testing()

add_subdirectory(src)
add_subdirectory(applications)
