# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

add_library(neofoam_catch_main_mpi test_main_mpi.cpp mpiReporter.cpp mpiGlobals.cpp
                                   mpiSerialization.cpp)
target_link_libraries(neofoam_catch_main_mpi PUBLIC Catch2::Catch2 cpptrace::cpptrace
                                                    Kokkos::kokkos MPI::MPI_CXX NeoFOAM)
target_link_libraries(neofoam_catch_main_mpi PRIVATE neofoam_warnings neofoam_options)

add_library(neofoam_catch_main test_main.cpp)
target_link_libraries(neofoam_catch_main PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos
                                                NeoFOAM)
target_link_libraries(neofoam_catch_main PRIVATE neofoam_warnings neofoam_options)
