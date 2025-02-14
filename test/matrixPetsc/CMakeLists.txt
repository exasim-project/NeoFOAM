# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

function(neofoam_unit_test_petsc TEST)
  set(options "")
  set(oneValueKeywords "COMMAND" "WORKING_DIRECTORY")
  set(multiValueKeywords "")
  cmake_parse_arguments("neofoam" "${options}" "${oneValueKeywords}" "${multiValueKeywords}"
                        ${ARGN})
  if(NOT DEFINED "neofoam_WORKING_DIRECTORY")
    set(neofoam_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()
  if(NOT DEFINED "neofoam_COMMAND")
    set(neofoam_COMMAND ${TEST})
  endif()

  add_definitions(${PETSc_DEFINITIONS})
  add_executable(${TEST} "${TEST}.cpp")
  target_link_libraries(
    ${TEST}
    PRIVATE ${PETSc_LIBRARIES}
            neofoam_catch_main
            neofoam_warnings
            neofoam_options
            Kokkos::kokkos
            NeoFOAM
            cpptrace::cpptrace
            ${PETSc_LIBRARIES}
            ${PETSc_LIBRARY_DIRS}/libkokkoscore.so.4.4)
  target_include_directories(${TEST} PRIVATE ${PETSc_INCLUDE_DIRS})
  target_link_directories(${TEST} PRIVATE ${PETSc_LIBRARY_DIRS})
  link_directories(${PETSc_LIBRARY_DIRS})
  set_target_properties(${TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${neofoam_WORKING_DIRECTORY})

  add_test(
    NAME ${TEST}
    COMMAND ${neofoam_COMMAND}
    WORKING_DIRECTORY ${neofoam_WORKING_DIRECTORY})

endfunction()

neofoam_unit_test_petsc(matrixAssembly)
