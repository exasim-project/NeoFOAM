# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

neofoam_unit_test(test_sundials)
target_link_libraries(
  test_sundials PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos Kokkos::kokkoskernels sundials_cvode
                        sundials_nvecserial)

add_executable(test_sun_kokkos "test_sun_kokkos.cpp" "test_sunlinsol.c")
target_link_libraries(
  test_sun_kokkos PRIVATE NeoFOAM Kokkos::kokkos Kokkos::kokkoskernels sundials_arkode
                          sundials_nvecmpiplusx sundials_nvecopenmpdev)

add_executable(test_sun_ark_heat "ark_heat2D.cpp")
target_link_libraries(test_sun_ark_heat PRIVATE NeoFOAM Kokkos::kokkos Kokkos::kokkoskernels
                                                sundials_arkode)
