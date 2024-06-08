# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME SurfaceField_test COMMAND test_SurfaceField)
add_executable(test_SurfaceField "cellCentred/fields/test_surfaceField.cpp"
                                 "../test_main.cpp")
target_link_libraries(test_SurfaceField PRIVATE Catch2::Catch2 NeoFOAM
                                                Kokkos::kokkos)

add_test(NAME VolummeField_test COMMAND test_VolumeField)
add_executable(test_VolumeField "cellCentred/fields/test_volumeField.cpp"
                                "../test_main.cpp")
target_link_libraries(test_VolumeField PRIVATE Catch2::Catch2 NeoFOAM
                                               Kokkos::kokkos)
