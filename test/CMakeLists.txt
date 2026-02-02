# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_subdirectory(catch2)

function(neofoam_unit_test TEST SETUP_DIRECTORY)
  if(NOT DEFINED "neofoam_WORKING_DIRECTORY")
    set(neofoam_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()

  add_executable(neofoam_${TEST} "test_${TEST}.cpp")

  target_compile_definitions(neofoam_${TEST} PUBLIC OMPI_SKIP_MPICXX)

  target_link_libraries(neofoam_${TEST} NeoFOAM_public_api neofoam_catch_main OpenFOAM)

  set(NFT_PWD ${neofoam_WORKING_DIRECTORY})
  set_target_properties(neofoam_${TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NFT_PWD})

  add_test(
    NAME neofoam_${TEST}
    COMMAND ${neofoam_WORKING_DIRECTORY}/neofoam_${TEST}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})

  install(TARGETS neofoam_${TEST})
endfunction()

neofoam_unit_test(readDict setup_operator)
neofoam_unit_test(unstructuredMesh setup_compatibility)
neofoam_unit_test(geometricFields setup_operator)
neofoam_unit_test(operators setup_operator)
neofoam_unit_test(stencils setup_backwardDdt)
neofoam_unit_test(sparsityPattern setup_backwardDdt)
neofoam_unit_test(implicitOperators setup_backwardDdt)
neofoam_unit_test(backwardDdtScheme setup_backwardDdt)
neofoam_unit_test(ddtFluxCorr setup_ddtCorr)
neofoam_unit_test(pressureVelocityCoupling setup_pressureVelocityCoupling)
neofoam_unit_test(advection setup_advection)
neofoam_unit_test(compatibility setup_compatibility)
neofoam_unit_test(momentum setup_pressureVelocityCoupling)
