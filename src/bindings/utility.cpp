// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/compatibility/fvSchemes.hpp"

// OpenFOAM headers
#include "fvCFD.H"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerUtility(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Utility: read a dimensioned scalar from OpenFOAM constant/ dict
    // -------------------------------------------------------------------
    m.def(
        "read_transport_viscosity",
        [](nf::RunTime& rt) -> double
        {
            Foam::IOdictionary transportProperties(Foam::IOobject(
                "transportProperties",
                rt.mesh.time().constant(),
                rt.mesh,
                Foam::IOobject::MUST_READ_IF_MODIFIED,
                Foam::IOobject::NO_WRITE
            ));
            Foam::dimensionedScalar nu("nu", Foam::dimViscosity, transportProperties);
            return nu.value();
        },
        "runtime"_a,
        "Read kinematic viscosity nu from constant/transportProperties"
    );

    m.def(
        "map_fv_solution",
        [](const NeoN::Dictionary& dict) { return nf::mapFvSolution(dict); },
        "dict"_a,
        "Map OpenFOAM solver names to Ginkgo equivalents"
    );

    m.def(
        "map_fv_schemes",
        [](const NeoN::Dictionary& dict) { return nf::mapFvSchemes(dict); },
        "dict"_a,
        "Map OpenFOAM scheme names to NeoN equivalents"
    );
}

} // namespace NeoFOAM::bindings
