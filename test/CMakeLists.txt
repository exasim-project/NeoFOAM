# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_subdirectory(catch2)

function(foam_adapter_unit_test TEST SETUP_DIRECTORY)
  if(NOT DEFINED "adapter_WORKING_DIRECTORY")
    set(adapter_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()

  add_executable(adapter_${TEST} "test_${TEST}.cpp")

  target_compile_definitions(adapter_${TEST} PUBLIC OMPI_SKIP_MPICXX)

  target_link_libraries(adapter_${TEST} NeoFOAM_public_api neofoam_catch_main OpenFOAM)

  set(NFT_PWD ${adapter_WORKING_DIRECTORY})
  set_target_properties(adapter_${TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NFT_PWD})

  add_test(
    NAME adapter_${TEST}
    COMMAND ${adapter_WORKING_DIRECTORY}/adapter_${TEST}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${SETUP_DIRECTORY})

  install(TARGETS adapter_${TEST})
endfunction()

foam_adapter_unit_test(geometricFields setup_operator)
foam_adapter_unit_test(operators setup_operator)
foam_adapter_unit_test(stencils setup_stencil3D)
foam_adapter_unit_test(sparsityPattern setup_stencil3D)
foam_adapter_unit_test(implicitOperators setup_stencil3D)
foam_adapter_unit_test(backwardDdtScheme setup_backwardDdt)
foam_adapter_unit_test(ddtFluxCorr setup_ddtCorr)
foam_adapter_unit_test(readDict setup_operator)
foam_adapter_unit_test(pressureVelocityCoupling setup_pressureVelocityCoupling)
foam_adapter_unit_test(unstructuredMesh setup_unstructuredMesh)
foam_adapter_unit_test(advection setup_advection)
foam_adapter_unit_test(compatibility setup_compatibility)
foam_adapter_unit_test(spalartAllmarasDDES setup_saddes)
