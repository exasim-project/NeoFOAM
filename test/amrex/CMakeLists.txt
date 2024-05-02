# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME amrex_test COMMAND test_amrex)
add_executable(test_amrex "test_amrex.cpp")
enable_language(CUDA)
setup_target_for_cuda_compilation(test_amrex)
target_link_libraries(test_amrex PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                         AMReX::amrex_2d)
