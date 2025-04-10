# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoN authors

neon_unit_test(CSRMatrix)
neon_unit_test(linearSystem)

# the following tests currently require Ginkgo
if(NeoN_WITH_GINKGO)
  neon_unit_test(ginkgo)
  neon_unit_test(linearAlgebra)
endif()
