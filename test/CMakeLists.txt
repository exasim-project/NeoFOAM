# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(test_field "test_Field.cpp" "test_main.cpp")
target_link_libraries(test_field PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

add_executable(test_executor "test_Executor.cpp" "test_main.cpp")
target_link_libraries(test_executor PRIVATE Catch2::Catch2 NeoFOAM
                                            Kokkos::kokkos)

add_executable(test_amrex "test_amrex.cpp")
# these are required to build test_amrex with cuda support
enable_language(CUDA)
setup_target_for_cuda_compilation(test_amrex)

target_link_libraries(test_amrex PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                         AMReX::amrex_2d)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
enable_testing()
add_test(NAME field_test COMMAND test_field)
add_test(NAME executor_test COMMAND test_executor)
add_test(NAME amrex_test COMMAND test_amrex)
