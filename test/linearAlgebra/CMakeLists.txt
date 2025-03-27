# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

neofoam_unit_test(CSRMatrix)
neofoam_unit_test(linearSystem)

# the following tests currently require Ginkgo
if(NEOFOAM_WITH_GINKGO)
  neofoam_unit_test(ginkgo)
  neofoam_unit_test(linearAlgebra)
endif()
