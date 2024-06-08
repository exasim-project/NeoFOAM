# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

add_test(NAME test_communicator COMMAND test_communicator)
add_executable(test_communicator "test_communicator.cpp" "../../test_main.cpp")
target_link_libraries(
  test_communicator PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                            ${MPI_CXX_LIBRARIES} cpptrace::cpptrace)
