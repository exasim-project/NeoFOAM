# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include_directories(${googletest_SOURCE_DIR}/googletest/include)
enable_testing()
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
