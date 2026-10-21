# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# has to be first since it adds the main target
add_subdirectory(catch2)

# Generate a system/controlDict for a test setup directory from the common stub at
# test/cmake/controlDict.in.  All parameters have sane defaults; only the values that differ from
# those defaults need to be listed.
#
# Usage: neofoam_generate_controlDict(<setup-dir> [APPLICATION   <name>]      # default:
# laplacianFoam [END_TIME      <value>]     # default: 1 [DELTA_T       <value>]     # default:
# 0.005 [WRITE_CONTROL <keyword>]   # default: runTime [WRITE_INTERVAL <value>]    # default: 10000
# [WRITE_FORMAT  <keyword>]   # default: ascii [WRITE_PRECISION <value>]   # default: 16
# [TIME_PRECISION <value>]    # default: 16 [START_FROM    <keyword>]   # default: startTime
# [EXECUTOR      <name>]      # optional (omitted when not set) [ALLOCATOR     <name>]      #
# optional (omitted when not set) [EXTRAS        <string>]    # optional raw OpenFOAM dict text
# appended # before the closing comment [PROFILING]                 # flag — adds a profiling{}
# sub-dict )
function(neofoam_generate_controlDict SETUP_DIR)
  set(_flags PROFILING)
  set(_one
      APPLICATION
      END_TIME
      DELTA_T
      WRITE_CONTROL
      WRITE_INTERVAL
      WRITE_FORMAT
      WRITE_PRECISION
      TIME_PRECISION
      START_FROM
      EXECUTOR
      ALLOCATOR)
  set(_multi EXTRAS)
  cmake_parse_arguments(CTRL "${_flags}" "${_one}" "${_multi}" ${ARGN})

  if(NOT DEFINED CTRL_APPLICATION)
    set(CTRL_APPLICATION "laplacianFoam")
  endif()
  if(NOT DEFINED CTRL_END_TIME)
    set(CTRL_END_TIME "1")
  endif()
  if(NOT DEFINED CTRL_DELTA_T)
    set(CTRL_DELTA_T "0.005")
  endif()
  if(NOT DEFINED CTRL_WRITE_CONTROL)
    set(CTRL_WRITE_CONTROL "runTime")
  endif()
  if(NOT DEFINED CTRL_WRITE_INTERVAL)
    set(CTRL_WRITE_INTERVAL "10000")
  endif()
  if(NOT DEFINED CTRL_WRITE_FORMAT)
    set(CTRL_WRITE_FORMAT "ascii")
  endif()
  if(NOT DEFINED CTRL_WRITE_PRECISION)
    set(CTRL_WRITE_PRECISION "16")
  endif()
  if(NOT DEFINED CTRL_TIME_PRECISION)
    set(CTRL_TIME_PRECISION "16")
  endif()
  if(NOT DEFINED CTRL_START_FROM)
    set(CTRL_START_FROM "startTime")
  endif()
  if(NOT DEFINED CTRL_EXTRAS)
    set(CTRL_EXTRAS "")
  else()
    # CMake strips ';' list separators (including trailing ones). Rejoin to restore terminators.
    list(JOIN CTRL_EXTRAS ";" CTRL_EXTRAS)
    string(APPEND CTRL_EXTRAS ";\n")
  endif()

  if(CTRL_EXECUTOR)
    set(CTRL_EXECUTOR_LINE "executor        ${CTRL_EXECUTOR};")
  else()
    set(CTRL_EXECUTOR_LINE "")
  endif()

  if(CTRL_ALLOCATOR)
    set(CTRL_ALLOCATOR_LINE "allocator       ${CTRL_ALLOCATOR};")
  else()
    set(CTRL_ALLOCATOR_LINE "")
  endif()

  if(CTRL_PROFILING)
    string(
      CONCAT CTRL_PROFILING_BLOCK
             "profiling\n{\n"
             "    active      true;\n"
             "    cpuInfo     true;\n"
             "    memInfo     true;\n"
             "    sysInfo     true;\n"
             "}\n")
  else()
    set(CTRL_PROFILING_BLOCK "")
  endif()

  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/controlDict.in"
                 "${CMAKE_SOURCE_DIR}/test/${SETUP_DIR}/system/controlDict" @ONLY)
endfunction()
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

# --- Generate controlDicts from the common stub ---

neofoam_generate_controldict(
  setup_operator
  APPLICATION
  laplacianFoam
  END_TIME
  3
  DELTA_T
  0.005
  PROFILING)

neofoam_generate_controldict(
  setup_backwardDdt
  APPLICATION
  laplacianFoam
  END_TIME
  3
  DELTA_T
  0.005
  PROFILING)

neofoam_generate_controldict(
  setup_compatibility
  APPLICATION
  laplacianFoam
  END_TIME
  3
  DELTA_T
  0.005
  PROFILING)

neofoam_generate_controldict(
  setup_pressureVelocityCoupling
  APPLICATION
  laplacianFoam
  END_TIME
  3
  DELTA_T
  0.01
  PROFILING)

neofoam_generate_controldict(
  setup_pressureVelocityCoupling_graded
  APPLICATION
  laplacianFoam
  END_TIME
  3
  DELTA_T
  1
  PROFILING)

neofoam_generate_controldict(
  setup_pressureVelocityCoupling_mpi2
  APPLICATION
  laplacianFoam
  END_TIME
  3
  DELTA_T
  1
  PROFILING)

neofoam_generate_controldict(
  setup_snGrad
  APPLICATION
  laplacianFoam
  END_TIME
  1
  DELTA_T
  1
  WRITE_CONTROL
  timeStep
  WRITE_INTERVAL
  1
  PROFILING)

neofoam_generate_controldict(
  setup_pimple
  APPLICATION
  neoPimpleFoam
  END_TIME
  2
  DELTA_T
  0.005
  WRITE_CONTROL
  timeStep
  WRITE_INTERVAL
  400
  WRITE_PRECISION
  15
  TIME_PRECISION
  6
  EXECUTOR
  Serial
  ALLOCATOR
  default)

neofoam_generate_controldict(
  setup_ddtCorr
  APPLICATION
  icoFoam
  START_FROM
  latestTime
  END_TIME
  0.0002
  DELTA_T
  0.0001
  WRITE_CONTROL
  timeStep
  WRITE_INTERVAL
  1
  WRITE_FORMAT
  binary
  WRITE_PRECISION
  6
  TIME_PRECISION
  6
  EXECUTOR
  GPU)

neofoam_generate_controldict(
  setup_advection
  APPLICATION
  scalarAdvection
  END_TIME
  3.0
  DELTA_T
  0.005
  TIME_PRECISION
  8
  WRITE_CONTROL
  none
  WRITE_INTERVAL
  3.0
  EXECUTOR
  Serial
  PROFILING
  EXTRAS
  "adjustTimeStep  yes;\n\nmaxCo           0.1;\n\nmaxDeltaT       1;\n\nsetFields       1;")

# --- Tests ---

# neofoam_unit_test(readDict setup_operator)
# neofoam_unit_test(unstructuredMesh setup_compatibility)
# neofoam_unit_test(geometricFields setup_operator)
# neofoam_unit_test(operators setup_operator)
# neofoam_unit_test(stencils setup_backwardDdt)
# neofoam_unit_test(implicitOperators setup_backwardDdt)
# neofoam_unit_test(backwardDdtScheme setup_backwardDdt)
# neofoam_unit_test(ddtFluxCorr setup_ddtCorr)
# neofoam_unit_test(setReference setup_ddtCorr)
# neofoam_unit_test(setConstraints setup_ddtCorr)
# neofoam_unit_test(pimpleControl setup_pimple)
# neofoam_unit_test(pimpleParity setup_pimple)
# neofoam_unit_test(pressureVelocityCoupling setup_pressureVelocityCoupling)
# neofoam_unit_test(advection setup_advection)
# neofoam_unit_test(compatibility setup_compatibility)
# neofoam_unit_test(bound setup_compatibility)
# neofoam_unit_test(momentum setup_pressureVelocityCoupling)
# neofoam_unit_test(snGrad setup_snGrad)
# neofoam_unit_test(residual setup_pressureVelocityCoupling)
# neofoam_unit_test(continuityError setup_pressureVelocityCoupling)
# neofoam_unit_test(forceCoeffs setup_pressureVelocityCoupling SOURCE test_forceCoeffs.cpp)

# add_subdirectory(distributed)
# add_subdirectory(turbulence)
