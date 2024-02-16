# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(test_blas "test_blas.cpp")
target_link_libraries(test_blas PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)
