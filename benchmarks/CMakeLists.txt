# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-25 NeoFOAM authors

function(neofoam_benchmark TEST)

  add_executable(bench_${TEST} "${TEST}.cpp")
  target_link_libraries(bench_${TEST} PRIVATE Catch2::Catch2 NeoFOAM)

  if(WIN32)
    set_target_properties(
      bench_${TEST}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/$<0:>
                 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/$<0:>
                 ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/$<0:>)
  else()
    set_property(TARGET bench_${TEST} PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()
  if(NOT DEFINED "neofoam_COMMAND")
    set(neofoam_COMMAND bench_${TEST})
  endif()
  if(NOT DEFINED "neofoam_WORKING_DIRECTORY")
    set(neofoam_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()
  add_test(
    NAME bench_${TEST}
    COMMAND ${neofoam_COMMAND}
    WORKING_DIRECTORY ${neofoam_WORKING_DIRECTORY})
endfunction()

add_subdirectory(fields)
