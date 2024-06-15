# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME SurfaceField_test COMMAND test_SurfaceField)
add_executable(test_SurfaceField "cellCentred/fields/surfaceField.cpp"
                                 "../test_main.cpp")
target_link_libraries(
  test_SurfaceField PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                            cpptrace::cpptrace)

add_test(NAME VolumeField_test COMMAND test_VolumeField)
add_executable(test_VolumeField "cellCentred/fields/volumeField.cpp"
                                "../test_main.cpp")
target_link_libraries(
  test_VolumeField PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                           cpptrace::cpptrace)

add_test(NAME boundaryField_test COMMAND test_boundaryField)
add_executable(
  test_boundaryField "cellCentred/fields/boundary/test_boundaryField.cpp"
                     "../test_main.cpp")
target_link_libraries(
  test_boundaryField PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                             cpptrace::cpptrace)
