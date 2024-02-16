# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# Add your test executable
add_executable(bench_blas "bench_blas.cpp")
target_link_libraries(bench_blas PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)