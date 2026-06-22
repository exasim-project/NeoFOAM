// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

// Explicit instantiation definitions for PDE<scalar> and PDE<Vec3>.
// All SYCL kernels referenced by PDE member functions (FixedValueConstraints,
// SetReference, dsl operators) are compiled here rather than in every translation
// unit that uses PDE — keeping per-TU kernel counts below Intel PVC AOT limits.

#include "NeoFOAM/datastructures/pde.hpp"

namespace NeoFOAM
{

template class PDE<NeoN::scalar>;
template class PDE<NeoN::Vec3>;

} // namespace NeoFOAM
