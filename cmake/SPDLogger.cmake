# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
include(FetchContent)

FetchContent_Declare(
  spdlogger
  QUITE
  GIT_SHALLOW ON
  GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
  GIT_TAG "v1.13.0")

FetchContent_MakeAvailable(spdlogger)

FetchContent_Declare(
  cxxopts
  QUITE
  GIT_SHALLOW ON
  GIT_REPOSITORY "https://github.com/jarro2783/cxxopts.git"
  GIT_TAG "v3.2.1")

FetchContent_MakeAvailable(cxxopts)
