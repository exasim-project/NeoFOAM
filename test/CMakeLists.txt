# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

if(NEOFOAM_BUILD_WITH_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0 # or a later release
  )
  FetchContent_MakeAvailable(Catch2)
endif()

add_executable(test_blas "test_blas.cpp")
target_link_libraries(test_blas PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)
