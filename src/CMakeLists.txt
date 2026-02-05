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
          # "datastructures/foamMesh.cpp"
          "datastructures/meshAdapter.cpp"
          "compatibility/fvSchemes.cpp"
          "compatibility/fvSolution.cpp")
if (NEOFOAM_ENABLE_PROFILING)
  target_sources(NeoFOAM PRIVATE auxiliary/profiling_kokkos.cpp)
  target_link_libraries(NeoFOAM PRIVATE Kokkos::kokkos)
else()
  target_sources(NeoFOAM PRIVATE auxiliary/profiling_none.cpp)
endif()
install(TARGETS NeoFOAM)
