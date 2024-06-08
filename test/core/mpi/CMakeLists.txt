# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_test(NAME test_mpi COMMAND test_mpi)
add_executable(test_mpi "test_mpi.cpp" "../../test_main.cpp")
target_link_libraries(test_mpi PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                       cpptrace::cpptrace ${MPI_CXX_LIBRARIES})
