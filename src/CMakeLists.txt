# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_subdirectory(NeoFOAM)
# add_subdirectory(DSL)

if(NEOFOAM_BUILD_WITH_TESTS)
  FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.4.0 # or a later release
  )
  FetchContent_MakeAvailable(Catch2)
  
  add_subdirectory(test)
  add_subdirectory(benchmarks)
endif()

