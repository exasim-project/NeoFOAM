# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_subdirectory(fields)

add_custom_command(
    TARGET execute_benchmarks
    COMMAND ${PROJECT_BINARY_DIR}/benchmarks/bench_blas -r XML > blas.xml
    COMMENT "Execute Benchmarks"
)
