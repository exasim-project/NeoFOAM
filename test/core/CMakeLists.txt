# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  add_subdirectory(mpi)
endif()

add_subdirectory(primitives)
add_subdirectory(database)

neofoam_unit_test(dictionary)
neofoam_unit_test(tokenList)
neofoam_unit_test(input)
neofoam_unit_test(executor)
neofoam_unit_test(parallelAlgorithms)

add_executable(runTimeSelectionFactory "runTimeSelectionFactory.cpp")
target_link_libraries(runTimeSelectionFactory PRIVATE Catch2::Catch2WithMain cpptrace::cpptrace
                                                      NeoFOAM_public_api)
if(WIN32)
  set_target_properties(
    runTimeSelectionFactory
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>)

else()
  set_property(TARGET runTimeSelectionFactory PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
endif()

add_test(
  NAME runTimeSelectionFactory
  COMMAND runTimeSelectionFactory
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
