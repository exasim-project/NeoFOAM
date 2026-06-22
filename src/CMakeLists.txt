# SPDX-License-Identifier: Unlicense
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM SHARED)

target_compile_definitions(NeoFOAM INTERFACE namespaceFoam=1)
target_link_libraries(NeoFOAM PUBLIC NeoFOAM_public_api OpenFOAM)

target_sources(
  NeoFOAM
  PRIVATE "algorithms/pressureVelocityCoupling.cpp"
          "auxiliary/convert.cpp"
          "auxiliary/foamDictionary.cpp"
          "auxiliary/setup.cpp"
          "auxiliary/writers.cpp"
          "auxiliary/comparison.cpp"
          "datastructures/meshAdapter.cpp"
          "datastructures/databaseWrapper.cpp"
          "datastructures/pde.cpp"
          "compatibility/fvSchemes.cpp"
          "compatibility/fvSolution.cpp"
          "solutionControl/pimpleControl.cpp"
          "turbulenceModels/turbulenceModel.cpp"
          "turbulenceModels/laminar.cpp"
          "turbulenceModels/spalartAllmarasDDES.cpp"
          "turbulenceModels/kOmegaSST.cpp"
          "turbulenceModels/kEpsilon.cpp"
          "fvcc/operators/viscousStressOperator.cpp"
          "functionObjects/functionObjectIO.cpp"
          "functionObjects/forces.cpp"
          "functionObjects/forceCoeffs.cpp")

install(TARGETS NeoFOAM)
