# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})

ExternalProject_Get_Property(petsc)

function(petsc_test TEST)
  add_executable(${TEST} "${TEST}.cpp")
  set_property(TARGET ${TEST} PROPERTY ENABLE_EXPORTS 1)
  target_include_directories(
    ${TEST}
    PUBLIC ${CMAKE_BINARY_DIR}/petsc/src/petsc/include ${CMAKE_BINARY_DIR}/petsc/opt/petsc/include/
           ${CMAKE_BINARY_DIR}/petsc/src/petsc/include/)
  link_directories(${CMAKE_BINARY_DIR}/petsc/opt/petsc/lib)
  target_link_libraries(${TEST} PUBLIC ${CMAKE_BINARY_DIR}/petsc/opt/petsc/lib/libpetsc.so)
  target_link_options(${TEST} PUBLIC -L${CMAKE_BINARY_DIR}/petsc/opt/petsc/lib)
  add_dependencies(${TEST} petsc)
endfunction()

petsc_test(vectorMatrixMultiplication)
