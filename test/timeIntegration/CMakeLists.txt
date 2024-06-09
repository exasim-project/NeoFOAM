# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

# add_test(NAME sundials_test COMMAND test_sundials)
# add_executable(test_sundials "test_sundials.cpp" "../test_main.cpp")
# target_link_libraries(test_sundials PRIVATE Catch2::Catch2 NeoFOAM
# Kokkos::kokkos)
add_test(NAME sundials_test COMMAND test_sundials)
add_executable(test_sundials "test_sundials.cpp" "../test_main.cpp")
target_link_libraries(
  test_sundials PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos sundials_cvode
                        sundials_nvecserial)
