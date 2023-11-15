# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(
  core_test
  "test_time.cpp"
)

target_link_libraries(
  core_test
  GTest::gtest_main NeoFOAM
)

include(GoogleTest)
gtest_discover_tests(core_test)
