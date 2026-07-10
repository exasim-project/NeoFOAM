# SPDX-License-Identifier: Unlicense
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM SHARED)

target_compile_definitions(NeoFOAM INTERFACE namespaceFoam=1)
target_link_libraries(NeoFOAM PUBLIC NeoFOAM_public_api OpenFOAM)

target_sources(
  NeoFOAM
  PRIVATE "algorithms/pressureVelocityCoupling.cpp"
          "algorithms/vofOperators.cpp"
          "auxiliary/convert.cpp"
          "auxiliary/foamDictionary.cpp"
          "auxiliary/setup.cpp"
          "auxiliary/writers.cpp"
          "auxiliary/comparison.cpp"
          "datastructures/meshAdapter.cpp"
          "datastructures/databaseWrapper.cpp"
          "compatibility/fvSchemes.cpp"
          "compatibility/fvSolution.cpp"
          "solutionControl/pimpleControl.cpp"
          "turbulenceModels/turbulenceModel.cpp"
          "turbulenceModels/laminar.cpp"
          "turbulenceModels/spalartAllmarasDDES.cpp"
          "fvcc/operators/viscousStressOperator.cpp"
          "functionObjects/functionObjectIO.cpp"
          "functionObjects/forces.cpp"
          "functionObjects/forceCoeffs.cpp")

# vofOperators.cpp mixes Kokkos device kernels with OpenFOAM headers (fvCFD.H). Without
# namespaceFoam=1 OpenFOAM injects sqrt/sin/... into the global namespace, which collide with CUDA's
# device-stub math (ambiguous overload). The NeoFOAM target only exports this define via INTERFACE
# (so consumers such as the bindings get it), so set it directly on this source. Namespacing-only
# macro — no effect on the algorithm's numerics.
set_property(SOURCE "algorithms/vofOperators.cpp" PROPERTY COMPILE_DEFINITIONS "namespaceFoam=1")

install(TARGETS NeoFOAM)
