# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

function(NeoFOAMUnitTest TEST)
  add_executable(${TEST} "${TEST}.cpp" "${PROJECT_SOURCE_DIR}/test/test_main.cpp")
  target_link_libraries(${TEST} PRIVATE Catch2::Catch2 NeoFOAM cpptrace::cpptrace)

  add_test(NAME ${TEST} COMMAND mpiexec -n 4 ${CMAKE_BINARY_DIR}/bin/${TEST})
  set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
endfunction()
