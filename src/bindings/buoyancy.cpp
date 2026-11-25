// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// The fvc quantities of a Boussinesq p_rgh formulation (buoyantBoussinesqPimpleFoam) that NeoN
// has no Python handle for: the geopotential gh/ghf of gh.H, the face areas |Sf| as a surface
// field, and snGrad with the case's snGrad scheme.

#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/algorithms/pressureVelocityCoupling.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

// OpenFOAM headers
#include "fvCFD.H"
#include "uniformDimensionedFields.H"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

namespace
{

// constant/g, as readGravitationalAcceleration.H reads it: a missing file is a FatalError there
// too, so a case with buoyancy must carry one.
Foam::uniformDimensionedVectorField readGravity(const nf::RunTime& rt)
{
    return Foam::uniformDimensionedVectorField(Foam::IOobject(
        "g",
        rt.mesh.time().constant(),
        rt.mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ));
}

// ghRef of readhRef.H + gh.H: constant/hRef (optional, default 0) projected on g's direction.
Foam::dimensionedScalar
readGhRef(const nf::RunTime& rt, const Foam::uniformDimensionedVectorField& g)
{
    Foam::uniformDimensionedScalarField hRef(
        Foam::IOobject(
            "hRef",
            rt.mesh.time().constant(),
            rt.mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        Foam::dimensionedScalar(Foam::word::null, Foam::dimLength, Foam::Zero)
    );
    if (Foam::mag(g.value()) > Foam::SMALL)
    {
        return g & (Foam::cmptMag(g.value()) / Foam::mag(g.value())) * hRef;
    }
    return Foam::dimensionedScalar("ghRef", g.dimensions() * Foam::dimLength, 0);
}

} // namespace

void registerBuoyancy(nb::module_& m)
{
    m.def(
        "create_gh",
        [](nf::RunTime& rt)
        {
            const auto g = readGravity(rt);
            const Foam::volScalarField gh("gh", (g & rt.mesh.C()) - readGhRef(rt, g));
            return nf::constructFrom(rt.exec, rt.nfMesh, gh);
        },
        "runtime"_a,
        "The geopotential gh = (g & C) - ghRef of OpenFOAM's gh.H, from constant/g and the "
        "optional constant/hRef; boundary values on the face centres"
    );

    m.def(
        "create_ghf",
        [](nf::RunTime& rt)
        {
            const auto g = readGravity(rt);
            const Foam::surfaceScalarField ghf("ghf", (g & rt.mesh.Cf()) - readGhRef(rt, g));
            return nf::constructFrom(rt.exec, rt.nfMesh, ghf);
        },
        "runtime"_a,
        "The face geopotential ghf = (g & Cf) - ghRef of OpenFOAM's gh.H"
    );

    m.def(
        "mag_sf",
        [](nf::RunTime& rt)
        {
            const Foam::surfaceScalarField magSf("magSf", rt.mesh.magSf());
            return nf::constructFrom(rt.exec, rt.nfMesh, magSf);
        },
        "runtime"_a,
        "Face-area magnitudes |Sf| as a scalar surface field, for face-flux arithmetic"
    );

    m.def(
        "sn_grad",
        [](nf::RunTime& rt, const fvcc::VolumeField<NeoN::scalar>& field)
        {
            if (&field.mesh() != &rt.nfMesh)
            {
                throw std::invalid_argument(
                    "sn_grad: field '" + field.name + "' lives on another mesh than the runtime"
                );
            }
            return nf::snGrad(field, rt.fvSchemesDict);
        },
        "runtime"_a,
        "field"_a,
        "Face-normal gradient fvc::snGrad(field) with the runtime's snGradSchemes entry "
        "(snGrad(<name>), else default): corrected/limitedCorrected include the "
        "non-orthogonal correction"
    );
}

} // namespace NeoFOAM::bindings
