# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 23)
project(minimal-example)

# We only need to find NeoFOAM if we build this example stand-alone
if(NOT NEOFOAM_BUILD_WITH_APPS)
  find_package(NeoFOAM CONFIG REQUIRED)
endif()

add_executable(minimalFoam "minimal.cpp")

target_link_libraries(minimalFoam PUBLIC NeoFOAM)
