# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include_directories(${googletest_SOURCE_DIR}/googletest/include)
enable_testing()
add_executable(core_test "test_time.cpp")

target_link_libraries(core_test GTest::gtest_main NeoFOAM)

include(GoogleTest)
gtest_discover_tests(core_test)

# add_executable(dsl_test "test_DSL.cpp")

# target_link_libraries(dsl_test GTest::gtest_main NeoFOAM DSL)

# include(GoogleTest)
# gtest_discover_tests(dsl_test)

add_executable(test_blas "test_blas_main.cpp")

target_link_libraries(test_blas GTest::gtest_main Kokkos::kokkos NeoFOAM)
include(GoogleTest)
gtest_discover_tests(test_blas)
