# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  add_subdirectory(mpi)
endif()

add_subdirectory(primitives)

neofoam_unit_test(dictionary)
neofoam_unit_test(tokenList)
neofoam_unit_test(inputs)
neofoam_unit_test(executor)
neofoam_unit_test(parallelAlgorithms)

# The registerClass test has to be created without linking against neofoam. Otherwise, it will pick
# up the factories registered in neofoam, which will make the test hard to maintain.
add_executable(runTimeSelectionFactory "runTimeSelectionFactory.cpp")
target_link_libraries(runTimeSelectionFactory PRIVATE Catch2::Catch2WithMain cpptrace::cpptrace
                                                      NeoFOAM_public_api)
add_test(
  NAME runTimeSelectionFactory
  COMMAND runTimeSelectionFactory
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
