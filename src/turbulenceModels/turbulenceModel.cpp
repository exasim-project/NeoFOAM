// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"

#include "IOdictionary.H"
#include "IOobject.H"

namespace NeoFOAM
{

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
        (simulationType == "laminar")
            ? "laminar"
            : std::string(turbProps.subDict("LES").get<Foam::word>("LESModel"));

    using Factory = NeoN::RuntimeSelectionFactory<
        TurbulenceModel,
        NeoN::Parameters<RunTime&, const nnfvcc::VolumeField<NeoN::scalar>&>>;
    return Factory::create(modelKey, rt, nu);
}

} // namespace NeoFOAM
