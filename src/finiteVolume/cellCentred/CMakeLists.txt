# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

target_sources(
  NeoFOAM
  PRIVATE "boundary/surface/scalar/empty.cpp"
          "boundary/surface/scalar/calculated.cpp"
          "boundary/volume/scalar/fixedValue.cpp"
          "boundary/volume/scalar/zeroGradient.cpp"
          "boundary/volume/scalar/empty.cpp"
          "boundary/volume/scalar/calculated.cpp"
          "boundary/volume/vector/fixedValue.cpp"
          "boundary/volume/vector/zeroGradient.cpp"
          "boundary/volume/vector/empty.cpp"
          "boundary/volume/vector/calculated.cpp"
          "boundarySelector.cpp")
