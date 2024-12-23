# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

neofoam_unit_test(timeIntegration)

if(WIN32)

else()
  neofoam_unit_test(rungeKutta)
endif()
