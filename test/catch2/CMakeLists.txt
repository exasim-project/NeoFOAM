# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

add_library(neofoam_catch_main test_main.cpp mpiReporter.cpp mpiGlobals.cpp mpiSerialization.cpp)
target_link_libraries(neofoam_catch_main PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos
                                                MPI::MPI_CXX NeoFOAM)
