# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

add_test(NAME test_communicator COMMAND test_communicator)
add_executable(test_communicator "test_communicator.cpp")
target_link_libraries(test_communicator PRIVATE neofoam_catch_main)
