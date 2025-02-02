# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

neofoam_unit_test(timeIntegration)
neofoam_unit_test(implicitTimeIntegration)
if(NOT WIN32)
  neofoam_unit_test(rungeKutta)
endif()
