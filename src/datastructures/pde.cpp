// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

// Non-template wrappers that call the FixedValueConstraints and SetReference SYCL kernels.
// Defined here so the kernels are compiled exactly once rather than in every translation unit
// that instantiates PDE<scalar>::solveImpl (e.g. kOmegaSST.cpp, kEpsilon.cpp, …), which
// would otherwise push those TUs over Intel PVC's per-TU AOT compilation limit.

#include "NeoFOAM/datastructures/pde.hpp"

namespace NeoFOAM::detail
{

void applyFixedValueConstraints(
    ScalarLinearSystem& ls,
    NeoN::View<const NeoN::scalar> mask,
    NeoN::View<const NeoN::scalar> values,
    NeoN::localIdx nCells
)
{
    NeoN::dsl::FixedValueConstraints<NeoN::scalar> pin(mask, values, nCells);
    pin(ls);
}

void applySetReference(ScalarLinearSystem& ls, NeoN::localIdx refCell, NeoN::scalar refValue)
{
    NeoN::dsl::SetReference<NeoN::scalar> refFunct(refCell, refValue);
    refFunct(ls);
}

} // namespace NeoFOAM::detail
