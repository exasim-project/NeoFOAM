# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_subdirectory(mpi)

add_test(NAME dictionary_test COMMAND test_dictionary)
add_executable(test_dictionary "test_Dictionary.cpp")
target_link_libraries(test_dictionary PRIVATE neofoam_catch_main)

add_test(NAME executor_test COMMAND test_executor)
add_executable(test_executor "test_Executor.cpp")
target_link_libraries(test_executor PRIVATE neofoam_catch_main)
