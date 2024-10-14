# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

# has to be first since it adds the main target
add_subdirectory(catch2)

# This function creates unit tests. It allows the following keywords:
#
# * MPI_SIZE: the number of MPI processors to be used, defaults to 1 if not set
# * COMMAND: the test command (same behavior as for CMake's add_test), defaults to the test name
# * WORKING_DIRECTORY: the working directory for the test, defaults to CMAKE_BINARY_DIR/bin
function(neofoam_unit_test TEST)
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
  if(NEOFOAM_ENABLE_CUDA)
    set_source_files_properties("${TEST}.cpp" PROPERTIES LANGUAGE CUDA)
  endif()
  target_link_libraries(${TEST} PRIVATE neofoam_catch_main neofoam_warnings neofoam_options
                                        Kokkos::kokkos NeoFOAM cpptrace::cpptrace)
  if(MSVC)
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

function(neofoam_unit_test_mpi TEST)
  set(options "")
  set(oneValueKeywords "MPI_SIZE" "COMMAND" "WORKING_DIRECTORY")
  set(multiValueKeywords "")
  cmake_parse_arguments("neofoam" "${options}" "${oneValueKeywords}" "${multiValueKeywords}"
                        ${ARGN})
  if(NOT DEFINED "neofoam_MPI_SIZE")
    set(neofoam_MPI_SIZE 1)
  endif()
  if(NOT DEFINED "neofoam_WORKING_DIRECTORY")
    set(neofoam_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()
  if(NOT DEFINED "neofoam_COMMAND")
    set(neofoam_COMMAND ${TEST})
  endif()

  add_executable(${TEST} "${TEST}.cpp")
  if(NEOFOAM_ENABLE_CUDA)
    set_source_files_properties("${TEST}.cpp" PROPERTIES LANGUAGE CUDA)
  endif()
  target_link_libraries(${TEST} PRIVATE neofoam_catch_main_mpi neofoam_warnings neofoam_options
                                        Kokkos::kokkos NeoFOAM cpptrace::cpptrace)
  if(MSVC)
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
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${neofoam_MPI_SIZE} ${neofoam_COMMAND}
    WORKING_DIRECTORY ${neofoam_WORKING_DIRECTORY})
  set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
  set_tests_properties(${TEST} PROPERTIES PROCESSORS ${neofoam_MPI_SIZE})
  if(${MPIEXEC_MAX_NUMPROCS} LESS neofoam_MPI_SIZE)
    message(WARNING "Test <${TEST}> requested <${neofoam_MPI_SIZE}> processors, "
                    "but only <${numberOfProcessors}> are available. Disabling test " "<${TEST}>")
    set_tests_properties(${TEST} PROPERTIES DISABLED TRUE)
  endif()
endfunction()

add_subdirectory(core)
add_subdirectory(fields)
add_subdirectory(finiteVolume)
add_subdirectory(mesh)
