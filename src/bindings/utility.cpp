// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

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
        "read_thermophysical_properties",
        [](nf::RunTime& rt) -> std::map<std::string, double>
        {
            Foam::IOdictionary thermoDict(Foam::IOobject(
                "thermophysicalProperties",
                rt.mesh.time().constant(),
                rt.mesh,
                Foam::IOobject::MUST_READ_IF_MODIFIED,
                Foam::IOobject::NO_WRITE
            ));
            const Foam::dictionary& mixture = thermoDict.subDict("mixture");
            const Foam::dictionary& specie = mixture.subDict("specie");
            const Foam::dictionary& thermodynamics = mixture.subDict("thermodynamics");
            const Foam::dictionary& transport = mixture.subDict("transport");

            std::map<std::string, double> out;
            out["molWeight"] = specie.get<double>("molWeight");
            out["mu"] = transport.get<double>("mu");
            out["Pr"] = transport.get<double>("Pr");
            if (thermodynamics.found("Cv"))
            {
                out["Cv"] = thermodynamics.get<double>("Cv");
            }
            if (thermodynamics.found("Cp"))
            {
                out["Cp"] = thermodynamics.get<double>("Cp");
            }
            out["Hf"] = thermodynamics.found("Hf") ? thermodynamics.get<double>("Hf") : 0.0;
            return out;
        },
        "runtime"_a,
        "Read molWeight/Cv-or-Cp/Hf/mu/Pr from constant/thermophysicalProperties "
        "(perfectGas + const transport)"
    );

    m.def(
        "map_fv_solution",
        [](const NeoN::Dictionary& dict) { return nf::mapFvSolution(dict); },
        "dict"_a,
        "Map OpenFOAM solver names to Ginkgo equivalents"
    );

    m.def(
        "map_solver_settings",
        &nf::mapSolverSettings,
        "solvers"_a,
        "field"_a,
        "Map the fvSolution solvers entry selected by field (OpenFOAM regex keys "
        "honoured) to its Ginkgo equivalent, in place"
    );

    // Surfaces as neofoam_bindings.FvSolutionKeyNotFound instead of the NeoN
    // dictionary's bare IndexError("unordered_map::at").
    nb::exception<nf::FvSolutionKeyNotFound>(m, "FvSolutionKeyNotFound", PyExc_RuntimeError);

    m.def(
        "map_fv_schemes",
        [](const NeoN::Dictionary& dict) { return nf::mapFvSchemes(dict); },
        "dict"_a,
        "Map OpenFOAM scheme names to NeoN equivalents"
    );
}

} // namespace NeoFOAM::bindings
