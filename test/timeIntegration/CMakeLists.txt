# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

neofoam_unit_test(test_sundials)
target_link_libraries(test_sundials PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos sundials_cvode
                                            sundials_nvecserial)
