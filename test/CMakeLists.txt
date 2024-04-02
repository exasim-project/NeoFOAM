# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(${PROJECT_SOURCE_DIR}/cmake/Catch2.cmake)

add_executable(test_field "fields/test_Field.cpp")
target_link_libraries(test_field PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
enable_testing()

add_test(NAME fields/field_test COMMAND test_field)

add_subdirectory(integration)
