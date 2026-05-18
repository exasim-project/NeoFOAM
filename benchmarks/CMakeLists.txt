# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-25 NeoN authors

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0)
FetchContent_MakeAvailable(Catch2)

set(Python_FIND_VIRTUALENV "FIRST")
find_package(Python COMPONENTS Interpreter)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

function(NeoN_benchmark BENCH)
  add_executable(bench_${BENCH} "${BENCH}.cpp")
  target_link_libraries(bench_${BENCH} PRIVATE Catch2::Catch2 OpenFOAM NeoFOAM)
  target_include_directories(bench_${BENCH} PRIVATE "${CMAKE_SOURCE_DIR}")
  if(NeoN_WITH_JULIA AND "${BENCH}" STREQUAL "julia")
    execute_process(
        COMMAND julia -e "print(rsplit(Sys.BINDIR, '/', limit=2)[1])"
        OUTPUT_VARIABLE JULIA_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND echo "[BENCHMARKING] JULIA_DIR: --------> '${JULIA_DIR}'"
    )
    target_include_directories(bench_${BENCH} PRIVATE "${JULIA_DIR}/include/julia")
    target_link_libraries(bench_${BENCH} PRIVATE
        "${JULIA_DIR}/lib/libjulia.so"
    )
    target_compile_definitions(bench_${BENCH} PRIVATE
        JULIA_MODULE_INIT="${JULIA_MODULE_INIT}"
    )
    target_link_options(bench_${BENCH} PRIVATE "-Wl,-rpath,${JULIA_DIR}/lib")
    target_compile_definitions(bench_${BENCH} PRIVATE
        JULIA_MODULE_INIT="${JULIA_MODULE_INIT}"
    )
  endif()

  if(WIN32)
    set_target_properties(
      bench_${BENCH}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/
                 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/
                 ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/)
  else()
    set_property(TARGET bench_${BENCH} PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()

  if(NOT DEFINED "NeoN_WORKING_DIRECTORY")
    set(NeoN_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()

  install(TARGETS bench_${BENCH})

  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmarkSuite/${BENCH}"
    DEPENDS bench_${BENCH}
    COMMENT "Generate ${BENCH} benchmark"
    COMMAND
      "${PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/benchmarks/benchmarkSuite/createStudies.py
      "generate" ${CMAKE_SOURCE_DIR}/benchmarks/benchmarkSuite ${CMAKE_BINARY_DIR}/benchmarkSuite
      ${BENCH} ${NEOFOAM_BENCHMARK_MODE})

  add_custom_target(
    benchmark_suite_${BENCH} ALL
    COMMENT "Target to generate test cases"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/benchmarkSuite/${BENCH}")
  add_test(
    NAME bench_${BENCH}
    COMMAND
      ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/benchmarks/benchmarkSuite/createStudies.py "execute"
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/bench_${BENCH} ${CMAKE_BINARY_DIR}/benchmarkSuite
      ${BENCH}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarkSuite)

endfunction()
if(NeoN_WITH_JULIA)
  neon_benchmark(julia)
endif()
neon_benchmark(explicitOperators)
neon_benchmark(implicitOperators)
neon_benchmark(scalarAdvection)
neon_benchmark(pressureVelocityCoupling)
neon_benchmark(momentum)
neon_benchmark(poisson)
  