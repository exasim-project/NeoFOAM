# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

# --- Generate controlDicts ---

neofoam_generate_controldict(
  turbulence/setup_kOmegaSST
  APPLICATION
  simpleFoam
  END_TIME
  1
  DELTA_T
  0.01)

neofoam_generate_controldict(
  turbulence/setup_kEpsilon
  APPLICATION
  simpleFoam
  END_TIME
  1
  DELTA_T
  0.01
  EXTRAS
  "libs    ( \"libturbulenceModels.so\" );")

neofoam_generate_controldict(
  turbulence/setup_epsilonWallFunction
  APPLICATION
  simpleFoam
  END_TIME
  1
  DELTA_T
  0.01
  EXTRAS
  "libs    ( \"libturbulenceModels.so\" );")

neofoam_generate_controldict(
  turbulence/setup_atmBoundaryLayer
  APPLICATION
  simpleFoam
  END_TIME
  1
  DELTA_T
  0.01
  EXTRAS
  "libs    ( \"libturbulenceModels.so\" \"libatmosphericModels.so\" );")

neofoam_generate_controldict(
  turbulence/setup_kEpsilonUndershoot
  APPLICATION
  simpleFoam
  END_TIME
  1
  DELTA_T
  1
  EXTRAS
  "libs    ( \"libturbulenceModels.so\" \"libatmosphericModels.so\" );")

# --- Tests ---

neofoam_unit_test(spalartAllmarasDDES setup_pressureVelocityCoupling SOURCE spalartAllmarasDDES.cpp)
neofoam_unit_test(kOmegaSST turbulence/setup_kOmegaSST SOURCE kOmegaSST.cpp)
neofoam_unit_test(kEpsilon turbulence/setup_kEpsilon SOURCE kEpsilon.cpp)
neofoam_unit_test(kqRWallFunction turbulence/setup_kEpsilon SOURCE kqRWallFunction.cpp)
neofoam_unit_test(omegaWallFunction turbulence/setup_kOmegaSST SOURCE omegaWallFunction.cpp)
neofoam_unit_test(epsilonWallFunction turbulence/setup_epsilonWallFunction SOURCE
                  epsilonWallFunction.cpp)
neofoam_unit_test(atmBoundaryLayer turbulence/setup_atmBoundaryLayer SOURCE atmBoundaryLayer.cpp)
neofoam_unit_test(kEpsilonUndershoot turbulence/setup_kEpsilonUndershoot SOURCE
                  kEpsilonUndershoot.cpp)
