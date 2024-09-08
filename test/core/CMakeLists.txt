# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_subdirectory(mpi)
add_subdirectory(primitives)

neofoam_unit_test(dictionary)
neofoam_unit_test(tokenList)
neofoam_unit_test(inputs)
neofoam_unit_test(executor)
neofoam_unit_test(parallelAlgorithms)

# The registerClass test has to be created without linking against neofoam. Otherwise, it will pick
# up the factories registered in neofoam, which will make the test hard to maintain.
add_executable(registerClass "registerClass.cpp")
target_link_libraries(registerClass PRIVATE Catch2::Catch2WithMain cpptrace::cpptrace MPI::MPI_CXX
                                            NeoFOAM_public_api)
add_test(
  NAME registerClass
  COMMAND registerClass
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
