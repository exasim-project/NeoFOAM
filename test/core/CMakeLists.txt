# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME dictionary_test COMMAND test_dictionary)
add_executable(test_dictionary "dictionary.cpp" "../test_main.cpp")
target_link_libraries(test_dictionary PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos
                                              cpptrace::cpptrace)

add_test(NAME executor_test COMMAND test_executor)
add_executable(test_executor "executor.cpp" "../test_main.cpp")
target_link_libraries(test_executor PRIVATE Catch2::Catch2 NeoFOAM Kokkos::kokkos)

add_test(NAME registerClass_test COMMAND test_registerClass)
add_executable(test_registerClass "registerClass.cpp" "../test_main.cpp")
target_link_libraries(test_registerClass PRIVATE Catch2::Catch2 NeoFOAM cpptrace::cpptrace
                                                 Kokkos::kokkos)
