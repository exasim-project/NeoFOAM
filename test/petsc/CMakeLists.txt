# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

# set(CMAKE_C_COMPILER ${PETSC_C_COMPILER}) set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})
set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
set(CMAKE_CXX_COMPILER nvcc_wrapper)

message("LIBRARY_DIRS")
message("${PETSc_LIBRARY_DIRS}")

message("INCLUDE_DIRS")
message("${PETSc_INCLUDE_DIRS}")

message("CMAKE_C_COMPILER")
message("${CMAKE_CXX_COMPILER}")

function(petsc_test TEST)
  include_directories(${PETSc_INCLUDE_DIRS})
  add_definitions(${PETSC_DEFINITIONS})
  add_executable(${TEST} "${TEST}.cpp")
  target_compile_options(${TEST} PRIVATE --extended-lambda -c -ccbin mpicxx)
  target_link_directories(${TEST} PRIVATE ${PETSc_LIBRARY_DIRS})
  # link_directories(${PETSc_LIBRARY_DIRS})
  target_link_libraries(${TEST} PRIVATE ${PETSc_LIBRARIES}
                                        ${PETSc_LIBRARY_DIRS}/libkokkoscore.so.4.4 mpi)
endfunction()

petsc_test(test_petsc_ksp)
petsc_test(vectorMatrixMultiplication)
petsc_test(kspKokkos)
