# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# has to be first since it adds the main target
add_subdirectory(catch2)
set(TEST_CATCH2_DIR ${CMAKE_CURRENT_LIST_DIR}/catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

function(neofoam_unit_test TEST SETUP_DIRECTORY)
  set(options "")
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

if(NEOFOAM_WITH_MPI)
  # Build executables and register the 3-rank CTest entries (default names without suffix).
  neofoam_unit_test(distributedUnstructuredMesh setup_pressureVelocityCoupling MPI_SIZE 3)
  neofoam_unit_test(distributedPressureVelocityCoupling setup_pressureVelocityCoupling MPI_SIZE 3)
  neofoam_unit_test(distributedMomentum setup_pressureVelocityCoupling MPI_SIZE 3)

  # 2-rank entries (ENABLED): use the 2-rank decomposition in setup_pressureVelocityCoupling_mpi2.
  foreach(TEST distributedUnstructuredMesh distributedPressureVelocityCoupling distributedMomentum)
    add_test(
      NAME ${TEST}_mpi2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/neofoam_test_${TEST} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/setup_pressureVelocityCoupling_mpi2)
    set_tests_properties(${TEST}_mpi2 PROPERTIES TIMEOUT 10 PROCESSORS 2)
  endforeach()

  # 4-rank entries (DISABLED): no 4-rank decomposition in Phase 2; enabled in Phase 5.
  foreach(TEST distributedUnstructuredMesh distributedPressureVelocityCoupling distributedMomentum)
    add_test(
      NAME ${TEST}_mpi4
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/neofoam_test_${TEST} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/setup_pressureVelocityCoupling)
    set_tests_properties(${TEST}_mpi4 PROPERTIES TIMEOUT 10 PROCESSORS 4 DISABLED TRUE)
  endforeach()

  # GPU-executor entries (DISABLED by default): run only the GPUExecutor path on GPU hardware.
  # These are identical to the 3-rank and 2-rank default entries except they filter Catch2
  # sections that contain "GPUExecutor" in their name.  Enable on GPU hardware with:
  #   ctest --preset develop -R "_gpu$" --overwrite DISABLED:BOOL=FALSE
  # Note: The 3-rank default entries (distributedUnstructuredMesh, etc.) and the _mpi2 entries
  # automatically run GPUExecutor sections when Kokkos::num_devices() > 0 at runtime.
  # The _gpu entries exist for selective GPU-only targeting during debugging.
  foreach(TEST distributedUnstructuredMesh distributedPressureVelocityCoupling distributedMomentum)
    add_test(
      NAME ${TEST}_gpu
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/neofoam_test_${TEST} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/setup_pressureVelocityCoupling)
    set_tests_properties(${TEST}_gpu PROPERTIES TIMEOUT 30 PROCESSORS 3 DISABLED TRUE)
  endforeach()

  foreach(TEST distributedUnstructuredMesh distributedPressureVelocityCoupling distributedMomentum)
    add_test(
      NAME ${TEST}_mpi2_gpu
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/neofoam_test_${TEST} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/setup_pressureVelocityCoupling_mpi2)
    set_tests_properties(${TEST}_mpi2_gpu PROPERTIES TIMEOUT 30 PROCESSORS 2 DISABLED TRUE)
  endforeach()

  # Graded-mesh entries (ENABLED): geometry correctness tests on non-uniform 2-rank mesh.
  # Only distributedUnstructuredMesh -- [GEO-01] and [GEO-02] are mesh-level tests
  # (no full pressure-velocity solve, per D-04 in 03-CONTEXT.md).
  foreach(TEST distributedUnstructuredMesh)
    add_test(
      NAME ${TEST}_graded
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/neofoam_test_${TEST} -parallel
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/setup_pressureVelocityCoupling_graded)
    set_tests_properties(${TEST}_graded PROPERTIES TIMEOUT 10 PROCESSORS 2)
  endforeach()

  # --------------------------------------------------------------------------- #
  # Doubly-graded toy reproducer for GPU multi-rank proc-boundary divergence
  # (track T2 in .planning/debug/gpu-proc-boundary-divergence.md).
  #
  # The toy case is NOT pre-decomposed in-tree (a committed polyMesh + processor{0,1}
  # tree would balloon the repo) — it ships only blockMeshDict + decomposeParDict
  # and is decomposed on first use via:
  #
  #   cd test/setup_distributedChannelToy_mpi2
  #   blockMesh && decomposePar -force
  #
  # The CTest entry is therefore registered DISABLED by default.  Enable on HPC
  # after decomposition with:
  #
  #   ctest --preset develop -R distributedChannelToy \
  #         --overwrite DISABLED:BOOL=FALSE
  #
  # The primary use of the toy case is to drive the `neoIcoFoam` binary itself
  # with NEOFOAM_PROC_DUMP=1 / NEOFOAM_FULL_DUMP=1 via tools/hpc-toy-bisect.sh;
  # this ctest entry is a 2-rank mesh-level sanity check (cell counts, proc patch).
  # --------------------------------------------------------------------------- #
  add_executable(neofoam_test_distributedChannelToy test_distributedChannelToy.cpp)
  install(TARGETS neofoam_test_distributedChannelToy)
  target_include_directories(neofoam_test_distributedChannelToy
                             PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(neofoam_test_distributedChannelToy
                        PROPERTIES OUTPUT_NAME neofoam_test_distributedChannelToy)
  target_compile_definitions(neofoam_test_distributedChannelToy
                             PUBLIC OMPI_SKIP_MPICXX)
  target_link_libraries(
    neofoam_test_distributedChannelToy
    PUBLIC NeoFOAM_public_api OpenFOAM Catch2::Catch2
    PRIVATE NeoN NeoFOAM neofoam_catch_main_mpi)
  set_target_properties(neofoam_test_distributedChannelToy
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)

  add_test(
    NAME distributedChannelToy_mpi2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/neofoam_test_distributedChannelToy
            -parallel
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/setup_distributedChannelToy_mpi2)
  set_tests_properties(distributedChannelToy_mpi2
                       PROPERTIES TIMEOUT 30 PROCESSORS 2 DISABLED TRUE)
endif()
