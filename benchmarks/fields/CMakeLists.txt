# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(bench_fields "bench_fields.cpp")
if(NEOFOAM_ENABLE_CUDA)
  set_source_files_properties("bench_fields.cpp" PROPERTIES LANGUAGE CUDA)
endif()
target_link_libraries(bench_fields PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

if(WIN32)
  set_target_properties(
    bench_fields
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/$<0:>)
else()
  set_property(TARGET bench_fields PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
endif()
