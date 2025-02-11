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

  add_executable(${TEST} "${TEST}.cpp")
  target_link_libraries(${TEST} PRIVATE neofoam_catch_main neofoam_warnings neofoam_options
                                        Kokkos::kokkos NeoFOAM cpptrace::cpptrace)

  if(NEOFOAM_WITH_SUNDIALS)
    target_link_libraries(${TEST} PRIVATE SUNDIALS::arkode)
  endif()
  if(NEOFOAM_WITH_PETSC)
    target_include_directories(${TEST} PRIVATE ${PETSc_INCLUDE_DIRS})
    link_directories(${PETSc_LIBRARY_DIRS})
    target_link_libraries(${TEST} PRIVATE petsc)
  endif()
  if(WIN32)
    set_target_properties(
      ${TEST}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${neofoam_WORKING_DIRECTORY}/$<0:>
                 LIBRARY_OUTPUT_DIRECTORY ${neofoam_WORKING_DIRECTORY}/$<0:>
                 ARCHIVE_OUTPUT_DIRECTORY ${neofoam_WORKING_DIRECTORY}/$<0:>)
  else()
    set_target_properties(${TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${neofoam_WORKING_DIRECTORY})
  endif()

  add_test(
    NAME ${TEST}
    COMMAND ${neofoam_COMMAND}
    WORKING_DIRECTORY ${neofoam_WORKING_DIRECTORY})

endfunction()

neofoam_unit_test_petsc(matrixAssembly)
