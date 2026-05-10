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
          "compatibility/fvSchemes.cpp"
          "compatibility/fvSolution.cpp")

install(TARGETS NeoFOAM)
