// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Explicit field under-relaxation used by the Python PIMPLE/SIMPLE loop: snapshot
// a field, blend solution = previous + alpha*(solution - previous) in place, and
// look up the per-field relaxation factor from fvSolution.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/dsl/solver.hpp" // fieldRelaxationSnapshot / applyFieldRelaxation

// NeoFOAM headers
#include "NeoFOAM/compatibility/fvSolution.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerFieldRelaxation(nb::module_& m)
{
    m.def(
        "field_relaxation_snapshot",
        [](const fvcc::VolumeField<NeoN::scalar>& field)
        { return NeoN::dsl::fieldRelaxationSnapshot(field); },
        "field"_a,
        "Deep-copy a field's internal vector for use as the relaxation reference"
    );

    m.def(
        "apply_field_relaxation",
        [](fvcc::VolumeField<NeoN::scalar>& solution,
           const NeoN::Vector<NeoN::scalar>& previous,
           double alpha) { NeoN::dsl::applyFieldRelaxation(solution, previous, alpha); },
        "solution"_a,
        "previous"_a,
        "alpha"_a,
        "Blend solution = previous + alpha*(solution - previous) in place"
    );

    m.def(
        "lookup_field_relaxation",
        [](const NeoN::Dictionary& fvSolution, const std::string& field, bool finalIter) -> double
        { return nf::lookupFieldRelaxation(fvSolution, field, finalIter).value_or(1.0); },
        "fv_solution"_a,
        "field"_a,
        "final_iter"_a,
        "Field under-relaxation factor from relaxationFactors.fields (1.0 if unset)"
    );
}

} // namespace NeoFOAM::bindings
