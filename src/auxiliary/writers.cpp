// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
//
#include "NeoFOAM/auxiliary/writers.hpp"

namespace NeoFOAM
{

void write(const NeoN::scalarVector& sf, const Foam::fvMesh& mesh, const std::string fieldName)
{
    Foam::volScalarField* field = mesh.getObjectPtr<Foam::volScalarField>(fieldName);
    if (field)
    {
        detail::copyImpl(sf, field->ref());
        field->write();
    }
    else
    {
        Foam::volScalarField foamField(
            Foam::IOobject(
                fieldName,
                mesh.time().timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar(Foam::dimless, 0)
        );
        detail::copyImpl(sf, foamField.ref());
        foamField.write();
    }
}

void write(
    const NeoN::Vector<NeoN::Vec3>& sf,
    const Foam::fvMesh& mesh,
    const std::string fieldName
)
{
    Foam::volVectorField* field = mesh.getObjectPtr<Foam::volVectorField>(fieldName);
    if (field)
    {
        // field is already present and needs to be updated
        detail::copyImpl(sf, field->ref());
        field->write();
    }
    else
    {
        Foam::volVectorField foamField(
            Foam::IOobject(
                fieldName,
                mesh.time().timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedVector(Foam::dimless, Foam::Zero)
        );
        detail::copyImpl(sf, foamField.ref());

        foamField.write();
    }
}

void write(const fvcc::VolumeField<NeoN::scalar>& volField, const Foam::fvMesh& mesh)
{
    Foam::volScalarField foamField(
        Foam::IOobject(
            volField.name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar(Foam::dimless, Foam::Zero)
    );
    detail::copyImpl(volField.internalVector(), foamField.ref());

    auto hostBCValue = volField.boundaryData().value().copyToHost();

    forAll(foamField.boundaryField(), patchi)
    {
        auto& foamFieldPatch = foamField.boundaryFieldRef()[patchi];
        auto [start, end] = volField.boundaryData().range(patchi);

        forAll(foamFieldPatch, bfacei)
        {
            foamFieldPatch[bfacei] = hostBCValue.view()[start + bfacei];
        }
    }
    foamField.write();
}

void write(const fvcc::VolumeField<NeoN::Vec3>& volField, const Foam::fvMesh& mesh)
{
    Foam::volVectorField foamField(
        Foam::IOobject(
            volField.name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedVector(Foam::dimless, Foam::Zero)
    );
    detail::copyImpl(volField.internalVector(), foamField.ref());

    auto hostBCValue = volField.boundaryData().value().copyToHost();

    forAll(foamField.boundaryField(), patchi)
    {
        auto& foamFieldPatch = foamField.boundaryFieldRef()[patchi];
        auto [start, end] = volField.boundaryData().range(patchi);

        forAll(foamFieldPatch, bfacei)
        {
            foamFieldPatch[bfacei][0] = hostBCValue.view()[start + bfacei][0];
            foamFieldPatch[bfacei][1] = hostBCValue.view()[start + bfacei][1];
            foamFieldPatch[bfacei][2] = hostBCValue.view()[start + bfacei][2];
        }
    }
    foamField.write();
}

}
