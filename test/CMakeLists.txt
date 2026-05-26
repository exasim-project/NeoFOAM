# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# has to be first since it adds the main target
add_subdirectory(catch2)
set(TEST_CATCH2_DIR ${CMAKE_CURRENT_LIST_DIR}/catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

function(neofoam_unit_test TEST SETUP_DIRECTORY)
  set(options "BLOCK_MESH")
  set(oneValueKeywords "MPI_SIZE" "COMMAND" "WORKING_DIRECTORY")
  set(multiValueKeywords "")
  cmake_parse_arguments("NEOFOAM" "${options}" "${oneValueKeywords}" "${multiValueKeywords}"
                        ${ARGN})
  if(NOT DEFINED "NEOFOAM_WORKING_DIRECTORY")
    set(NEOFOAM_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()
  if(NOT DEFINED "NEOFOAM_COMMAND")
    set(NEOFOAM_COMMAND neofoam_test_${TEST})
  else()
    message(Warning "NEOFOAM_command already defined ${NEOFOAM_COMMAND}")
  endif()

  add_executable("${NEOFOAM_COMMAND}" "test_${TEST}.cpp")
  install(TARGETS ${NEOFOAM_COMMAND})

  target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(${NEOFOAM_COMMAND} PROPERTIES OUTPUT_NAME neofoam_test_${TEST})
  target_compile_definitions(${NEOFOAM_COMMAND} PUBLIC OMPI_SKIP_MPICXX)

  target_link_libraries(
    ${NEOFOAM_COMMAND}
    PUBLIC NeoFOAM_public_api OpenFOAM Catch2::Catch2
    PRIVATE NeoN NeoFOAM)

  if(NOT NEOFOAM_MPI_SIZE)
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE neofoam_catch_main)
  else()
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE neofoam_catch_main_mpi)
  endif()

  if(WIN32)
    set_target_properties(
      ${NEOFOAM_COMMAND}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NEOFOAM_WORKING_DIRECTORY}/$<0:>
                 LIBRARY_OUTPUT_DIRECTORY ${NEOFOAM_WORKING_DIRECTORY}/$<0:>
                 ARCHIVE_OUTPUT_DIRECTORY ${NEOFOAM_WORKING_DIRECTORY}/$<0:>)
  else()
    # cmake-lint: disable=C0307
    set_target_properties(${NEOFOAM_COMMAND} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                        ${NEOFOAM_WORKING_DIRECTORY})
  endif()

  if(NOT NEOFOAM_MPI_SIZE)
    add_test(
      NAME neofoam_test_${TEST}
      COMMAND ${NEOFOAM_COMMAND}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})

    if(NEOFOAM_BLOCK_MESH)
      if(NOT NEOFOAM_BLOCKMESH_EXECUTABLE)
        find_program(NEOFOAM_BLOCKMESH_EXECUTABLE blockMesh
                     HINTS ENV FOAM_APPBIN ENV FOAM_USER_APPBIN
                     DOC "OpenFOAM blockMesh utility")
        if(NOT NEOFOAM_BLOCKMESH_EXECUTABLE)
          message(FATAL_ERROR
            "blockMesh not found — source OpenFOAM before configuring "
            "(required for test '${TEST}' which generates its polyMesh on the fly)")
        endif()
      endif()

      add_test(
        NAME neofoam_blockMesh_${TEST}
        COMMAND ${NEOFOAM_BLOCKMESH_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})
      set_tests_properties(neofoam_blockMesh_${TEST}
                           PROPERTIES FIXTURES_SETUP blockMesh_${TEST})
      set_tests_properties(neofoam_test_${TEST}
                           PROPERTIES FIXTURES_REQUIRED blockMesh_${TEST})
    endif()
  else()
    add_test(
      NAME ${TEST}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NEOFOAM_MPI_SIZE}
              ${NEOFOAM_WORKING_DIRECTORY}/${NEOFOAM_COMMAND} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})
    set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
    set_tests_properties(${TEST} PROPERTIES PROCESSORS ${NEOFOAM_MPI_SIZE})
    if(${MPIEXEC_MAX_NUMPROCS} LESS NEOFOAM_MPI_SIZE)
      message(WARNING "Test <${TEST}> requested <${NEOFOAM_MPI_SIZE}> processors, "
                      "but only <${MPIEXEC_MAX_NUMPROCS}> are available. Disabling test "
                      "<${TEST}>")
    endif()
  endif()
endfunction()

neofoam_unit_test(readDict setup_operator)
neofoam_unit_test(unstructuredMesh setup_compatibility)
neofoam_unit_test(geometricFields setup_operator)
neofoam_unit_test(operators setup_operator)
neofoam_unit_test(stencils setup_backwardDdt)
neofoam_unit_test(implicitOperators setup_backwardDdt)
neofoam_unit_test(backwardDdtScheme setup_backwardDdt)
neofoam_unit_test(ddtFluxCorr setup_ddtCorr)
neofoam_unit_test(pressureVelocityCoupling setup_pressureVelocityCoupling)
neofoam_unit_test(advection setup_advection)
neofoam_unit_test(compatibility setup_compatibility)
neofoam_unit_test(momentum setup_pressureVelocityCoupling)
neofoam_unit_test(snGrad setup_snGrad BLOCK_MESH)

if(NEOFOAM_WITH_MPI)
  neofoam_unit_test(distributedMomentum setup_pressureVelocityCoupling MPI_SIZE 3)
endif()
