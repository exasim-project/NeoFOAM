// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"
#include "NeoFOAM/turbulenceModels/laminar.hpp"
#include "NeoFOAM/turbulenceModels/spalartAllmarasDDES.hpp"
#include "NeoFOAM/fvcc/boundary/volume/nutWallFunction.hpp"

#include "IOdictionary.H"
#include "IOobject.H"

namespace NeoFOAM
{

std::unique_ptr<TurbulenceModel>
createTurbulenceModel(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu)
{
    // Defensive force-registration into libNeoFOAM's runtime-selection tables (see
    // makeViscousStress for the rationale): the automatic static self-registration is
    // elided when these types are only reached through the -fvisibility=hidden Python
    // bindings. The nutUSpaldingWallFunction boundary condition is otherwise never
    // ODR-used in libNeoFOAM (it is purely runtime-selected when SA-DDES reads nut),
    // so its registration would never fire. addSubType() is idempotent.
    TurbulenceModel::Register<Laminar>::addSubType();
    TurbulenceModel::Register<SpalartAllmarasDDES>::addSubType();
    nnfvcc::VolumeBoundaryFactory<NeoN::scalar>::Register<
        nnfvcc::volumeBoundary::NutUSpaldingWallFunction>::addSubType();
    return TurbulenceModel::create(rt, nu);
}

std::unique_ptr<TurbulenceModel>
TurbulenceModel::create(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu)
{
    Foam::IOdictionary turbProps(Foam::IOobject(
        "turbulenceProperties",
        rt.mesh.time().constant(),
        rt.mesh,
        Foam::IOobject::MUST_READ_IF_MODIFIED,
        Foam::IOobject::NO_WRITE
    ));

    const Foam::word simulationType(turbProps.get<Foam::word>("simulationType"));

    const std::string modelKey =
        (simulationType == "laminar") ? "laminar"
        : (simulationType == "RAS")
            ? std::string(turbProps.subDict("RAS").get<Foam::word>("RASModel"))
            : std::string(turbProps.subDict("LES").get<Foam::word>("LESModel"));

    using Factory = NeoN::RuntimeSelectionFactory<
        TurbulenceModel,
        NeoN::Parameters<RunTime&, const nnfvcc::VolumeField<NeoN::scalar>&>>;
    return Factory::create(modelKey, rt, nu);
}

} // namespace NeoFOAM
