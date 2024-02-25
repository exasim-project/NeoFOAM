# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

FetchContent_Declare(
Catch2
GIT_REPOSITORY https://github.com/catchorg/Catch2.git
GIT_TAG v3.4.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_blas "test_Field.cpp")
target_link_libraries(test_blas PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)
