# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(cmake/CPM.cmake)

cpmaddpackage(
  NAME
  amrex_code
  GITHUB_REPOSITORY
  AMReX-Codes/amrex
  VERSION
  24.04
  GIT_TAG
  24.04)

cpmaddpackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json VERSION 3.11.3)

cpmaddpackage(NAME spdlog GITHUB_REPOSITORY gabime/spdlog VERSION 1.13.0)

cpmaddpackage(NAME cxxopts GITHUB_REPOSITORY jarro2783/cxxopts VERSION 3.2.0)

if(NEOFOAM_BUILD_TESTS OR NEOFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(NAME Catch2 GITHUB_REPOSITORY catchorg/Catch2 VERSION 3.4.0)
endif()
