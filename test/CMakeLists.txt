# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# has to be first since it adds the main target
add_subdirectory(catch2)
set(TEST_CATCH2_DIR ${CMAKE_CURRENT_LIST_DIR}/catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

function(neofoam_unit_test TEST SETUP_DIRECTORY)
  set(options "")
  set(oneValueKeywords "MPI_SIZE" "COMMAND" "WORKING_DIRECTORY" "SOURCE")
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
  if(NOT DEFINED "NEOFOAM_SOURCE")
    set(NEOFOAM_SOURCE "${TEST}.cpp")
  endif()

  add_executable("${NEOFOAM_COMMAND}" "${NEOFOAM_SOURCE}")
  install(TARGETS ${NEOFOAM_COMMAND})

  target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(${NEOFOAM_COMMAND} PROPERTIES OUTPUT_NAME neofoam_test_${TEST})
  target_compile_definitions(${NEOFOAM_COMMAND} PUBLIC OMPI_SKIP_MPICXX)

  target_link_libraries(
    ${NEOFOAM_COMMAND}
    PUBLIC NeoFOAM_public_api OpenFOAM Catch2::Catch2
    PRIVATE NeoN NeoFOAM)

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
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE neofoam_catch_main)
    add_test(
      NAME neofoam_test_${TEST}
      COMMAND ${NEOFOAM_COMMAND}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})
  else()
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE neofoam_catch_main_mpi)
    add_test(
      NAME neofoam_test_${TEST}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NEOFOAM_MPI_SIZE}
              ${NEOFOAM_WORKING_DIRECTORY}/${NEOFOAM_COMMAND} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})
    set_tests_properties(neofoam_test_${TEST} PROPERTIES TIMEOUT 10)
    set_tests_properties(neofoam_test_${TEST} PROPERTIES PROCESSORS ${NEOFOAM_MPI_SIZE})
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
neofoam_unit_test(setReference setup_ddtCorr)
neofoam_unit_test(pimpleControl setup_pimple)
neofoam_unit_test(pimpleParity setup_pimple)
neofoam_unit_test(pressureVelocityCoupling setup_pressureVelocityCoupling)
neofoam_unit_test(advection setup_advection)
neofoam_unit_test(compatibility setup_compatibility)
neofoam_unit_test(momentum setup_pressureVelocityCoupling)
neofoam_unit_test(snGrad setup_snGrad)
neofoam_unit_test(residual setup_pressureVelocityCoupling)
neofoam_unit_test(spalartAllmarasDDES setup_pressureVelocityCoupling)
neofoam_unit_test(deShybrid setup_pressureVelocityCoupling)
neofoam_unit_test(continuityError setup_pressureVelocityCoupling)
neofoam_unit_test(forceCoeffs setup_pressureVelocityCoupling SOURCE test_forceCoeffs.cpp)

add_subdirectory(distributed)
