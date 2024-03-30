# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(${PROJECT_SOURCE_DIR}/cmake/Catch2.cmake)

add_executable(test_field "test_Field.cpp")
target_link_libraries(test_field PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)
enable_testing()
add_test(NAME field_test COMMAND test_field)


add_executable(test_dictionary "test_dictionary.cpp")
target_link_libraries(test_dictionary PRIVATE Catch2::Catch2WithMain NeoFOAM)
add_test(NAME dictionary_test COMMAND test_dictionary)
