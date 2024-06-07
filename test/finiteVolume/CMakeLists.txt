# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME finiteVolume_test COMMAND test_finiteVolume)
add_executable(test_finiteVolume "cellCentred/fields/test_surfaceField.cpp"
                                 "../test_main.cpp")
target_link_libraries(test_finiteVolume PRIVATE Catch2::Catch2 NeoFOAM
                                                Kokkos::kokkos)
