# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

neofoam_unit_test(spalartAllmarasDDES setup_pressureVelocityCoupling SOURCE spalartAllmarasDDES.cpp)
neofoam_unit_test(kOmegaSST turbulence/setup_kOmegaSST SOURCE kOmegaSST.cpp)
neofoam_unit_test(kEpsilon turbulence/setup_kEpsilon SOURCE kEpsilon.cpp)
neofoam_unit_test(kqRWallFunction turbulence/setup_kEpsilon SOURCE kqRWallFunction.cpp)
neofoam_unit_test(omegaWallFunction turbulence/setup_kOmegaSST SOURCE omegaWallFunction.cpp)
