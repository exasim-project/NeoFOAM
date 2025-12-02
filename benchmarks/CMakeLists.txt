# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-25 NeoN authors

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0)
FetchContent_MakeAvailable(Catch2)

function(NeoN_benchmark BENCH)

  add_executable(bench_${BENCH} "bench_${BENCH}.cpp")
  target_link_libraries(bench_${BENCH} PRIVATE Catch2::Catch2 OpenFOAM NeoFOAM)
  target_include_directories(bench_${BENCH} PRIVATE "${CMAKE_SOURCE_DIR}")

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
  add_test(
    NAME bench_${BENCH}
    COMMAND sh -c "./bench_${BENCH} -r xml > ${BENCH}.xml"
    WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY}/benchmarkSuite/${BENCH})
  install(TARGETS bench_${BENCH})
endfunction()

neon_benchmark(explicitOperators)
neon_benchmark(implicitOperators)
neon_benchmark(dsl)
