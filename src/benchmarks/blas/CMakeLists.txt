# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)

FetchContent_MakeAvailable(googlebenchmark)

find_package(Kokkos REQUIRED)

add_executable(bench_blas bench_blas.cpp)
target_link_libraries(bench_blas benchmark::benchmark NeoFOAM Kokkos::kokkos)
