# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_subdirectory(fields)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/benchmarks/fields.xml
  COMMAND
    ${PROJECT_BINARY_DIR}/benchmarks/fields/${CMAKE_BUILD_TYPE}/bench_fields -r
    XML > fields.xml
  COMMENT "Execute benchmarks")

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/benchmarks/fields.png
  COMMAND python ${PROJECT_SOURCE_DIR}/scripts/plotBenchmarks.py
          /benchmarks/fields/${CMAKE_BUILD_TYPE}/fields.xml
  COMMENT "Plot benchmark results")

add_custom_target(
  execute_benchmarks
  DEPENDS ${PROJECT_BINARY_DIR}/benchmarks/fields.xml
  COMMENT "execute benchmarks")
add_custom_target(
  execute_plot_benchmark
  DEPENDS ${PROJECT_BINARY_DIR}/benchmarks/fields.png
  COMMENT "plot benchmark results")
