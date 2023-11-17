# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_subdirectory(NeoFOAM)

if(NEOFOAM_BUILD_TESTS)
  add_subdirectory(test)
endif()
