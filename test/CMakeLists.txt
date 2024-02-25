# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(${PROJECT_SOURCE_DIR}/cmake/Catch2.cmake)
add_executable(test_blas "test_Field.cpp")
target_link_libraries(test_blas PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)
