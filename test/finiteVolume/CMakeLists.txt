# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME SurfaceField_test COMMAND test_SurfaceField)
add_executable(test_SurfaceField "cellCentred/fields/surfaceField.cpp" "../test_main.cpp")
target_link_libraries(test_SurfaceField PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                                cpptrace::cpptrace)

add_test(NAME VolumeField_test COMMAND test_VolumeField)
add_executable(test_VolumeField "cellCentred/fields/volumeField.cpp" "../test_main.cpp")
target_link_libraries(test_VolumeField PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                               cpptrace::cpptrace)

add_test(NAME fixedValue_test COMMAND test_fixedValue)
add_executable(test_fixedValue "cellCentred/boundary/fixedValue.cpp" "../test_main.cpp")
target_link_libraries(test_fixedValue PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                              cpptrace::cpptrace)

add_test(NAME fixedGradient_test COMMAND test_fixedGradient)
add_executable(test_fixedGradient "cellCentred/boundary/fixedGradient.cpp" "../test_main.cpp")
target_link_libraries(test_fixedGradient PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                                 cpptrace::cpptrace)
