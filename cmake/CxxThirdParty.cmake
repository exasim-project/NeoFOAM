# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(cmake/CPM.cmake)

CPMAddPackage("gh:kokkos/kokkos#4.3.00")

CPMAddPackage("gh:nlohmann/json@3.11.3")
CPMAddPackage("gh:gabime/spdlog@1.13.0")

CPMAddPackage(
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 3.2.0
  OPTIONS
  "CXXOPTS_BUILD_EXAMPLES NO"
  "CXXOPTS_BUILD_TESTS NO"
  "CXXOPTS_ENABLE_INSTALL YES"
  )

if(NEOFOAM_BUILD_TESTS)
    CPMAddPackage("gh:catchorg/Catch2@3.4.0")
endif()
