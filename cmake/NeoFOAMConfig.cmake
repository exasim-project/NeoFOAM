# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

function(NeoFOAMUnitTest TEST)    
	add_executable(${TEST} "${TEST}.cpp" "${PROJECT_SOURCE_DIR}/test/test_main.cpp")    
  target_link_libraries(${TEST}    
	  PRIVATE 
	Catch2::Catch2 NeoFOAM
	cpptrace::cpptrace
      )    
    
  add_test(NAME ${TEST} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST})    
endfunction()   
