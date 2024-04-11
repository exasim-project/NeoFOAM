# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(test_field "test_Field.cpp")
target_link_libraries(test_field PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

add_executable(test_executor "test_Executor.cpp")
target_link_libraries(test_executor PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
enable_testing()
add_test(NAME field_test COMMAND test_field)
add_test(NAME executor_test COMMAND test_executor)