# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME field_mesh COMMAND test_mesh)
add_executable(test_mesh "unstructuredMesh.cpp" "../test_main.cpp")
target_link_libraries(test_mesh PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)
