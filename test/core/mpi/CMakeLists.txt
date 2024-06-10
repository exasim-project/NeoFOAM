# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_test(NAME test_operators COMMAND test_operators)
add_executable(test_operators "test_operators.cpp" "../../test_main.cpp")
target_link_libraries(
  test_operators PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                         cpptrace::cpptrace)
