# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})

message("LIBRARY_DIRS")
message("${PETSc_LIBRARY_DIRS}")

message("INCLUDE_DIRS")
message("${PETSc_INCLUDE_DIRS}")

message("CMAKE_C_COMPILER")
message("${CMAKE_CXX_COMPILER}")

function(petsc_test TEST)
  add_executable(${TEST} "${TEST}.cpp")
  target_include_directories(${TEST} PRIVATE ${PETSc_INCLUDE_DIRS})
  link_directories(${PETSc_LIBRARY_DIRS})
  target_link_libraries(${TEST} PRIVATE petsc)
endfunction()

petsc_test(test_petsc_ksp)
petsc_test(vectorMatrixMultiplication)
