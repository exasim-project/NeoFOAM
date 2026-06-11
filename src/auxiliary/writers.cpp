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
    // Prefer an already-registered OF field (preserves original BC types such as fixedValue/
    // zeroGradient). If the caller registered ofP/ofU in the mesh object registry (without
    // NO_REGISTER), the getObjectPtr path is taken and the BC structure is correct.
    // Fall back: read from the start-time directory (READ_IF_PRESENT) to recover the original
    // BC types written by decomposePar / case setup. Without this the IOobject defaults all
    // boundary patches to 'calculated', erasing the pressure-reference fixedValue BC and making
    // the pressure equation singular on restart.
    Foam::volScalarField* regField = mesh.getObjectPtr<Foam::volScalarField>(volField.name);
    if (regField)
    {
        detail::copyImpl(volField.internalVector(), regField->ref());
        auto hostBCValue = volField.boundaryData().value().copyToHost();
        forAll(regField->boundaryField(), patchi)
        {
            auto& foamFieldPatch = regField->boundaryFieldRef()[patchi];
            [[maybe_unused]] auto [start, end] = volField.boundaryData().range(patchi);
            forAll(foamFieldPatch, bfacei)
            {
                foamFieldPatch[bfacei] = hostBCValue.view()[start + bfacei];
            }
        }
        regField->write();
        return;
    }

    // Not registered — read from start-time directory to obtain the original BC types.
    // READ_IF_PRESENT: if the file exists in startTime (e.g. "0/p"), the BC types are
    // preserved; if not (e.g. non-zero startTime and no file), defaults are used.
    Foam::volScalarField foamField(
        Foam::IOobject(
            volField.name,
            mesh.time().timeName(mesh.time().startTime().value()),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar(Foam::dimless, Foam::Zero)
    );
    foamField.writeOpt(Foam::IOobject::AUTO_WRITE);
    foamField.rename(volField.name);

    detail::copyImpl(volField.internalVector(), foamField.ref());

    auto hostBCValue = volField.boundaryData().value().copyToHost();

    forAll(foamField.boundaryField(), patchi)
    {
        auto& foamFieldPatch = foamField.boundaryFieldRef()[patchi];
        [[maybe_unused]] auto [start, end] = volField.boundaryData().range(patchi);

        forAll(foamFieldPatch, bfacei)
        {
            foamFieldPatch[bfacei] = hostBCValue.view()[start + bfacei];
        }
    }
    foamField.write();
}

void write(const fvcc::VolumeField<NeoN::Vec3>& volField, const Foam::fvMesh& mesh)
{
    // Same BC-preservation strategy as write(VolumeField<scalar>, mesh) above.
    Foam::volVectorField* regField = mesh.getObjectPtr<Foam::volVectorField>(volField.name);
    if (regField)
    {
        detail::copyImpl(volField.internalVector(), regField->ref());
        auto hostBCValue = volField.boundaryData().value().copyToHost();
        forAll(regField->boundaryField(), patchi)
        {
            auto& foamFieldPatch = regField->boundaryFieldRef()[patchi];
            [[maybe_unused]] auto [start, end] = volField.boundaryData().range(patchi);
            forAll(foamFieldPatch, bfacei)
            {
                foamFieldPatch[bfacei][0] = hostBCValue.view()[start + bfacei][0];
                foamFieldPatch[bfacei][1] = hostBCValue.view()[start + bfacei][1];
                foamFieldPatch[bfacei][2] = hostBCValue.view()[start + bfacei][2];
            }
        }
        regField->write();
        return;
    }

    Foam::volVectorField foamField(
        Foam::IOobject(
            volField.name,
            mesh.time().timeName(mesh.time().startTime().value()),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedVector(Foam::dimless, Foam::Zero)
    );
    foamField.writeOpt(Foam::IOobject::AUTO_WRITE);
    foamField.rename(volField.name);

    detail::copyImpl(volField.internalVector(), foamField.ref());

    auto hostBCValue = volField.boundaryData().value().copyToHost();

    forAll(foamField.boundaryField(), patchi)
    {
        auto& foamFieldPatch = foamField.boundaryFieldRef()[patchi];
        [[maybe_unused]] auto [start, end] = volField.boundaryData().range(patchi);

        forAll(foamFieldPatch, bfacei)
        {
            foamFieldPatch[bfacei][0] = hostBCValue.view()[start + bfacei][0];
            foamFieldPatch[bfacei][1] = hostBCValue.view()[start + bfacei][1];
            foamFieldPatch[bfacei][2] = hostBCValue.view()[start + bfacei][2];
        }
    }
    foamField.write();
}

void write(const fvcc::SurfaceField<NeoN::scalar>& surfField, const Foam::fvMesh& mesh)
{
    Foam::surfaceScalarField foamField(
        Foam::IOobject(
            surfField.name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar(Foam::dimless, Foam::Zero)
    );
    // split storage: internalVector() is nInternalFaces; primitiveFieldRef() is the same size
    detail::copyImpl(surfField.internalVector(), foamField.primitiveFieldRef());

    auto hostBCValue = surfField.boundaryData().value().copyToHost();

    forAll(foamField.boundaryField(), patchi)
    {
        auto& foamFieldPatch = foamField.boundaryFieldRef()[patchi];
        [[maybe_unused]] auto [start, end] = surfField.boundaryData().range(patchi);

        forAll(foamFieldPatch, bfacei)
        {
            foamFieldPatch[bfacei] = hostBCValue.view()[start + bfacei];
        }
    }
    foamField.write();
}

void write(const fvcc::SurfaceField<NeoN::Vec3>& surfField, const Foam::fvMesh& mesh)
{
    Foam::surfaceVectorField foamField(
        Foam::IOobject(
            surfField.name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedVector(Foam::dimless, Foam::Zero)
    );
    // split storage: internalVector() is nInternalFaces; primitiveFieldRef() is the same size
    detail::copyImpl(surfField.internalVector(), foamField.primitiveFieldRef());

    auto hostBCValue = surfField.boundaryData().value().copyToHost();

    forAll(foamField.boundaryField(), patchi)
    {
        auto& foamFieldPatch = foamField.boundaryFieldRef()[patchi];
        [[maybe_unused]] auto [start, end] = surfField.boundaryData().range(patchi);

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
