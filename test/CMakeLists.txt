# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# has to be first since it adds the main target
add_subdirectory(catch2)
set(TEST_CATCH2_DIR ${CMAKE_CURRENT_LIST_DIR}/catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
set(CASES_BASEDIR /storage/home/lupeterm/nobackup/julia/view/base/mesh)
set(DRIVAER /storage/home/lupeterm/nobackup/driveAerLukas/occDrivAerStaticMesh)
set(WINDSOR /storage/home/lupeterm/micro_benchmarks/WindsorBody/newn/workspace/89a6498ecf7fdef5df609579aa93098f/case)
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
    set(NEOFOAM_COMMAND neofoam_test_${TEST}_${SETUP_DIRECTORY})
  else()
    message(Warning "NEOFOAM_command already defined ${NEOFOAM_COMMAND}")
  endif()
  if(NOT DEFINED "NEOFOAM_SOURCE")
    set(NEOFOAM_SOURCE "test_${TEST}.cpp")
  endif()

  add_executable("${NEOFOAM_COMMAND}" "${NEOFOAM_SOURCE}")
  install(TARGETS ${NEOFOAM_COMMAND})

  target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(${NEOFOAM_COMMAND} PROPERTIES OUTPUT_NAME neofoam_test_${TEST}_${SETUP_DIRECTORY})
  target_compile_definitions(${NEOFOAM_COMMAND} PUBLIC OMPI_SKIP_MPICXX)

  if(NeoN_WITH_JULIA)

    execute_process(
      COMMAND julialauncher +1.11 -e "print(rsplit(Sys.BINDIR, '/', limit=2)[1])"
      OUTPUT_VARIABLE JULIA_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND echo "JULIA_DIR: --------> '${JULIA_DIR}'"
    )
    target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${JULIA_DIR}/include/julia")
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE
        "${JULIA_DIR}/lib/libjulia.so"
    )
    target_compile_definitions(${NEOFOAM_COMMAND} PRIVATE
        JUNE_MODULE_INIT="${JUNE_MODULE_INIT}"
    )
    target_link_options(${NEOFOAM_COMMAND} PRIVATE "-Wl,-rpath,${JULIA_DIR}/lib")
  endif()

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
      NAME neofoam_test_${TEST}_${SETUP_DIRECTORY}
      COMMAND ${NEOFOAM_COMMAND}
      WORKING_DIRECTORY ${CASES_BASEDIR}/${SETUP_DIRECTORY}/case)
  else()
    add_test(
      NAME ${TEST}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NEOFOAM_MPI_SIZE}
              ${NEOFOAM_WORKING_DIRECTORY}/${NEOFOAM_COMMAND} -parallel
      WORKING_DIRECTORY ${CASES_BASEDIR}/${SETUP_DIRECTORY}/case)
    set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
    set_tests_properties(${TEST} PROPERTIES PROCESSORS ${NEOFOAM_MPI_SIZE})
    if(${MPIEXEC_MAX_NUMPROCS} LESS NEOFOAM_MPI_SIZE)
      message(WARNING "Test <${TEST}> requested <${NEOFOAM_MPI_SIZE}> processors, "
                      "but only <${MPIEXEC_MAX_NUMPROCS}> are available. Disabling test "
                      "<${TEST}>")
    endif()
  endif()
endfunction()

function(drivaer TEST SETUP_DIRECTORY)
  set(options "")
  set(oneValueKeywords "MPI_SIZE" "COMMAND" "WORKING_DIRECTORY")
  set(multiValueKeywords "")
  cmake_parse_arguments("NEOFOAM" "${options}" "${oneValueKeywords}" "${multiValueKeywords}"
                        ${ARGN})
  if(NOT DEFINED "NEOFOAM_WORKING_DIRECTORY")
    set(NEOFOAM_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()
  if(NOT DEFINED "NEOFOAM_COMMAND")
    set(NEOFOAM_COMMAND neofoam_test_drivaer)
  else()
    message(Warning "NEOFOAM_command already defined ${NEOFOAM_COMMAND}")
  endif()

  add_executable("${NEOFOAM_COMMAND}" "test_drivaer.cpp")
  install(TARGETS ${NEOFOAM_COMMAND})

  target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(${NEOFOAM_COMMAND} PROPERTIES OUTPUT_NAME drivaer)
  target_compile_definitions(${NEOFOAM_COMMAND} PUBLIC OMPI_SKIP_MPICXX)

  if(NeoN_WITH_JULIA)

    execute_process(
      COMMAND julialauncher +1.11 -e "print(rsplit(Sys.BINDIR, '/', limit=2)[1])"
      OUTPUT_VARIABLE JULIA_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND echo "JULIA_DIR: --------> '${JULIA_DIR}'"
    )
    target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${JULIA_DIR}/include/julia")
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE
        "${JULIA_DIR}/lib/libjulia.so"
    )
    target_compile_definitions(${NEOFOAM_COMMAND} PRIVATE
        JUNE_MODULE_INIT="${JUNE_MODULE_INIT}"
    )
    target_link_options(${NEOFOAM_COMMAND} PRIVATE "-Wl,-rpath,${JULIA_DIR}/lib")
  endif()

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
      NAME drivaer
      COMMAND ${NEOFOAM_COMMAND}
      WORKING_DIRECTORY ${SETUP_DIRECTORY})
  else()
    add_test(
      NAME drivaer
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NEOFOAM_MPI_SIZE}
              ${NEOFOAM_WORKING_DIRECTORY}/${NEOFOAM_COMMAND} -parallel
      WORKING_DIRECTORY ${SETUP_DIRECTORY})
    set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
    set_tests_properties(${TEST} PROPERTIES PROCESSORS ${NEOFOAM_MPI_SIZE})
    if(${MPIEXEC_MAX_NUMPROCS} LESS NEOFOAM_MPI_SIZE)
      message(WARNING "Test <drivaer> requested <${NEOFOAM_MPI_SIZE}> processors, "
                      "but only <${MPIEXEC_MAX_NUMPROCS}> are available. Disabling test "
                      "<drivaer>")
    endif()
  endif()
endfunction()


function(windsor TEST SETUP_DIRECTORY)
  set(options "")
  set(oneValueKeywords "MPI_SIZE" "COMMAND" "WORKING_DIRECTORY")
  set(multiValueKeywords "")
  cmake_parse_arguments("NEOFOAM" "${options}" "${oneValueKeywords}" "${multiValueKeywords}"
                        ${ARGN})
  if(NOT DEFINED "NEOFOAM_WORKING_DIRECTORY")
    set(NEOFOAM_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()
  if(NOT DEFINED "NEOFOAM_COMMAND")
    set(NEOFOAM_COMMAND neofoam_test_windsor)
  else()
    message(Warning "NEOFOAM_command already defined ${NEOFOAM_COMMAND}")
  endif()

  add_executable("${NEOFOAM_COMMAND}" "test_strats.cpp")
  install(TARGETS ${NEOFOAM_COMMAND})

  target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(${NEOFOAM_COMMAND} PROPERTIES OUTPUT_NAME windsor)
  target_compile_definitions(${NEOFOAM_COMMAND} PUBLIC OMPI_SKIP_MPICXX)

  if(NeoN_WITH_JULIA)

    execute_process(
      COMMAND julialauncher +1.11 -e "print(rsplit(Sys.BINDIR, '/', limit=2)[1])"
      OUTPUT_VARIABLE JULIA_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND echo "JULIA_DIR: --------> '${JULIA_DIR}'"
    )
    target_include_directories(${NEOFOAM_COMMAND} PRIVATE "${JULIA_DIR}/include/julia")
    target_link_libraries(${NEOFOAM_COMMAND} PRIVATE
        "${JULIA_DIR}/lib/libjulia.so"
    )
    target_compile_definitions(${NEOFOAM_COMMAND} PRIVATE
        JUNE_MODULE_INIT="${JUNE_MODULE_INIT}"
    )
    target_link_options(${NEOFOAM_COMMAND} PRIVATE "-Wl,-rpath,${JULIA_DIR}/lib")
  endif()

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
      NAME windsor
      COMMAND ${NEOFOAM_COMMAND}
      WORKING_DIRECTORY ${SETUP_DIRECTORY})
  else()
    add_test(
      NAME windsor
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NEOFOAM_MPI_SIZE}
              ${NEOFOAM_WORKING_DIRECTORY}/${NEOFOAM_COMMAND} -parallel
      WORKING_DIRECTORY ${SETUP_DIRECTORY})
    set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
    set_tests_properties(${TEST} PROPERTIES PROCESSORS ${NEOFOAM_MPI_SIZE})
    if(${MPIEXEC_MAX_NUMPROCS} LESS NEOFOAM_MPI_SIZE)
      message(WARNING "Test <windsor> requested <${NEOFOAM_MPI_SIZE}> processors, "
                      "but only <${MPIEXEC_MAX_NUMPROCS}> are available. Disabling test "
                      "<windsor>")
    endif()
  endif()
endfunction()


# neofoam_unit_test(unstructuredMesh setup_compatibility)
# neofoam_unit_test(strats 200)


if(NeoN_WITH_JULIA)

  drivaer(drivaer ${DRIVAER})
  windsor(drivaer ${WINDSOR})
  # windsor(fused ${CASES_BASEDIR}/10/case)
  # neofoam_unit_test(fused 20)
  # neofoam_unit_test(fused 30)
  # neofoam_unit_test(fused 40)
  # neofoam_unit_test(fused 50)
  # neofoam_unit_test(fused 60)
  # neofoam_unit_test(fused 70)
  # neofoam_unit_test(fused 80)
  # neofoam_unit_test(fused 90)
  # neofoam_unit_test(fused 100)
  # neofoam_unit_test(fused 110)
  # neofoam_unit_test(fused 120)
  # neofoam_unit_test(fused 130)
  # neofoam_unit_test(fused 140)
  # neofoam_unit_test(fused 150)
  # neofoam_unit_test(fused 160)
  # neofoam_unit_test(fused 170)
  # neofoam_unit_test(fused 180)
  # neofoam_unit_test(fused 190)
  # neofoam_unit_test(fused 200)
  # neofoam_unit_test(fused 220)
  # neofoam_unit_test(fused 240)
  # neofoam_unit_test(fused 260)
  # neofoam_unit_test(fused 280)
  # neofoam_unit_test(fused 300)
  # neofoam_unit_test(fused 320)
  # neofoam_unit_test(fused 340)
  # neofoam_unit_test(fused 360)
  # neofoam_unit_test(fused 380)
  # neofoam_unit_test(drivaer 400)

  # neofoam_unit_test(julia 10)
  # neofoam_unit_test(julia 20)
  # neofoam_unit_test(julia 30)
  # neofoam_unit_test(julia 40)
  # neofoam_unit_test(julia 50)
  # neofoam_unit_test(julia 60)
  # neofoam_unit_test(julia 70)
  # neofoam_unit_test(julia 80)
  # neofoam_unit_test(julia 90)
  # neofoam_unit_test(julia 100)
  # neofoam_unit_test(julia 110)
  # neofoam_unit_test(julia 120)
  # neofoam_unit_test(julia 130)
  # neofoam_unit_test(julia 140)
  # neofoam_unit_test(julia 150)
  # neofoam_unit_test(julia 160)
  # neofoam_unit_test(julia 170)
  # neofoam_unit_test(julia 180)
  # neofoam_unit_test(julia 190)
  # neofoam_unit_test(julia 200)
  # neofoam_unit_test(julia 220)
  # neofoam_unit_test(julia 240)
  # neofoam_unit_test(julia 260)
  # neofoam_unit_test(julia 280)
  # neofoam_unit_test(julia 300)
  # neofoam_unit_test(julia 320)
  # neofoam_unit_test(julia 340)
  # neofoam_unit_test(julia 360)
  # neofoam_unit_test(julia 380)
  # neofoam_unit_test(julia 400)
endif()

# neofoam_unit_test(readDict setup_operator)
# neofoam_unit_test(unstructuredMesh setup_compatibility)
# neofoam_unit_test(geometricFields setup_operator)
# neofoam_unit_test(operators setup_operator)
# neofoam_unit_test(stencils setup_backwardDdt)
# neofoam_unit_test(implicitOperators setup_backwardDdt)
# neofoam_unit_test(backwardDdtScheme setup_backwardDdt)
# neofoam_unit_test(ddtFluxCorr setup_ddtCorr)
# neofoam_unit_test(setReference setup_ddtCorr)
# neofoam_unit_test(pimpleControl setup_pimple)
# neofoam_unit_test(pimpleParity setup_pimple)
# neofoam_unit_test(pressureVelocityCoupling setup_pressureVelocityCoupling)
# neofoam_unit_test(advection setup_advection)
# neofoam_unit_test(compatibility setup_compatibility)
# neofoam_unit_test(momentum setup_pressureVelocityCoupling)
# neofoam_unit_test(snGrad setup_snGrad)
# neofoam_unit_test(residual setup_pressureVelocityCoupling)
# neofoam_unit_test(continuityError setup_pressureVelocityCoupling)
# neofoam_unit_test(forceCoeffs setup_pressureVelocityCoupling SOURCE forceCoeffs.cpp)

# add_subdirectory(distributed)
# add_subdirectory(turbulence)
