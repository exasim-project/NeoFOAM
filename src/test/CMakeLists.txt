# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(
  core_test
  "test_hello_world.cpp"
)

target_link_libraries(
  core_test
  GTest::gtest_main NeoFOAM
)

include(GoogleTest)
gtest_discover_tests(core_test)
