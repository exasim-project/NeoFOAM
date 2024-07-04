# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_test(NAME test_fullDuplexCommBuffer COMMAND test_fullDuplexCommBuffer)
add_executable(test_fullDuplexCommBuffer "test_fullDuplexCommBuffer.cpp"
                                         "../../test_main.cpp")
target_link_libraries(test_fullDuplexCommBuffer
                      PRIVATE Catch2::Catch2 cpptrace::cpptrace NeoFOAM)

add_test(NAME test_halfDuplexCommBuffer COMMAND test_halfDuplexCommBuffer)
add_executable(test_halfDuplexCommBuffer "test_halfDuplexCommBuffer.cpp"
                                         "../../test_main.cpp")
target_link_libraries(test_halfDuplexCommBuffer
                      PRIVATE Catch2::Catch2 cpptrace::cpptrace NeoFOAM)
