// SPDX-FileCopyrightText: 2024-2026 NeoFOAM authors
// SPDX-License-Identifier: Unlicense

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/auxiliary/readers.hpp"

// OpenFOAM headers
#include "fvCFD.H"

#include "runtime.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerFieldFactories(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Field factories — read OF fields, return NeoN fields registered in db
    // -------------------------------------------------------------------
    m.def(
        "read_scalar_volume_field",
        [](Runtime& rt, const std::string& name) -> fvcc::VolumeField<NeoN::scalar>&
        {
            auto& nfrt = rt.nfRuntime();
            Foam::volScalarField ofField(
                Foam::IOobject(
                    name,
                    rt.foamTime().timeName(),
                    rt.mesh(),
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh()
            );
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(nfrt.db, "VectorCollection");
            return vc.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = nfrt.exec,
                    .nfMesh = nfrt.nfMesh,
                    .foamField = ofField,
                    .name = name
                }
            );
        },
        "runtime"_a,
        "name"_a,
        nb::rv_policy::reference,
        "Read an OpenFOAM scalar field and register it as a NeoN VolumeField"
    );

    m.def(
        "read_vector_volume_field",
        [](Runtime& rt, const std::string& name) -> fvcc::VolumeField<NeoN::Vec3>&
        {
            auto& nfrt = rt.nfRuntime();
            Foam::volVectorField ofField(
                Foam::IOobject(
                    name,
                    rt.foamTime().timeName(),
                    rt.mesh(),
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh()
            );
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(nfrt.db, "VectorCollection");
            return vc.registerVector<fvcc::VolumeField<NeoN::Vec3>>(
                nf::CreateFromFoamField<Foam::volVectorField> {
                    .exec = nfrt.exec,
                    .nfMesh = nfrt.nfMesh,
                    .foamField = ofField,
                    .name = name
                }
            );
        },
        "runtime"_a,
        "name"_a,
        nb::rv_policy::reference,
        "Read an OpenFOAM vector field and register it as a NeoN VolumeField"
    );

    m.def(
        "create_phi",
        [](Runtime& rt, const std::string& uFieldName) -> fvcc::SurfaceField<NeoN::scalar>&
        {
            auto& nfrt = rt.nfRuntime();
            Foam::volVectorField ofU(
                Foam::IOobject(
                    uFieldName,
                    rt.foamTime().timeName(),
                    rt.mesh(),
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh()
            );
            Foam::surfaceScalarField ofPhi(
                Foam::IOobject(
                    "phi",
                    rt.foamTime().timeName(),
                    rt.mesh(),
                    Foam::IOobject::READ_IF_PRESENT,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                Foam::fvc::flux(ofU)
            );
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(nfrt.db, "VectorCollection");
            return vc.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::surfaceScalarField> {
                    .exec = nfrt.exec,
                    .nfMesh = nfrt.nfMesh,
                    .foamField = ofPhi,
                    .name = std::string("phi")
                }
            );
        },
        "runtime"_a,
        "u_field_name"_a = std::string("U"),
        nb::rv_policy::reference,
        "Create phi (face flux) SurfaceField registered in VectorCollection"
    );

    m.def(
        "create_uniform_surface_field",
        [](Runtime& rt, const std::string& name, double value)
        {
            auto& nfrt = rt.nfRuntime();
            auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(nfrt.nfMesh);
            fvcc::SurfaceField<NeoN::scalar> field(nfrt.exec, name, nfrt.nfMesh, bcs);
            NeoN::fill(field.internalVector(), value);
            NeoN::fill(field.boundaryData().value(), value);
            return field;
        },
        "runtime"_a,
        "name"_a,
        "value"_a,
        "Create a uniform scalar surface field (e.g. viscosity)"
    );

    m.def(
        "set_ref_cell",
        [](Runtime& rt, const std::string& fieldName, const std::string& pisoDict
        ) -> std::tuple<int, double, bool>
        {
            // Read OF field temporarily just to check needReference
            Foam::volScalarField ofField(
                Foam::IOobject(
                    fieldName,
                    rt.foamTime().timeName(),
                    rt.mesh(),
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh()
            );
            Foam::label refCell = 0;
            Foam::scalar refValue = 0.0;
            Foam::setRefCell(
                ofField,
                rt.mesh().solutionDict().subDict(pisoDict),
                refCell,
                refValue
            );
            bool needs = ofField.needReference() && refCell >= 0;
            return {static_cast<int>(refCell), refValue, needs};
        },
        "runtime"_a,
        "field_name"_a = std::string("p"),
        "piso_dict"_a = std::string("PISO"),
        "Get (refCell, refValue, needsReference) from fvSolution"
    );
}

} // namespace NeoFOAM::bindings
