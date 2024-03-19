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
