// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h> // copy_from_host (host array -> field)
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/processor.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/processor.hpp"

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/fvcc/boundary/volume/epsilonWallFunction.hpp"
#include "NeoFOAM/fvcc/boundary/volume/kqRWallFunction.hpp"
#include "NeoFOAM/fvcc/boundary/volume/nutkWallFunction.hpp"
#include "NeoFOAM/fvcc/boundary/volume/omegaWallFunction.hpp"
#include "NeoFOAM/fvcc/boundary/volume/nutWallFunction.hpp"

// OpenFOAM headers
#include "fvCFD.H"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

// Force CRTP self-registration of the turbulence wall-function volume BCs into
// libNeoN's VolumeBoundaryFactory<scalar> runtime-selection table, without
// modifying the NeoN submodule. Each header defines a self-registering
// Register<>::REGISTERED static (its initialiser calls addSubType()); taking its
// address ODR-uses it in this TU, which is compiled into neofoam_bindings and
// links libNeoN, so the initialiser runs at module load and inserts the creator
// keyed by name() (e.g. "epsilonWallFunction") into the shared singleton table.
namespace
{
namespace vb = NeoN::finiteVolume::cellCentred::volumeBoundary;
using ScalarVolBCFactory = NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>;

[[maybe_unused]] const bool* const registerWallFunctionBCs[] = {
    &ScalarVolBCFactory::Register<vb::EpsilonWallFunction>::REGISTERED,
    &ScalarVolBCFactory::Register<vb::KqRWallFunction>::REGISTERED,
    &ScalarVolBCFactory::Register<vb::NutkWallFunction>::REGISTERED,
    &ScalarVolBCFactory::Register<vb::OmegaWallFunction>::REGISTERED,
    &ScalarVolBCFactory::Register<vb::NutUSpaldingWallFunction>::REGISTERED,
};

// Same workaround for NeoN's own processor (halo-exchange) BCs: their libNeoN
// instantiation registers them only in libNeoN's table copy, so a decomposed
// case read through this TU aborts with "Could not find constructor for
// processor" without these.
namespace sb = NeoN::finiteVolume::cellCentred::surfaceBoundary;
using VectorVolBCFactory = NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::Vec3>;
using ScalarSurfBCFactory = NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::scalar>;

[[maybe_unused]] const bool* const registerProcessorBCs[] = {
    &ScalarVolBCFactory::Register<vb::Processor<NeoN::scalar>>::REGISTERED,
    &VectorVolBCFactory::Register<vb::Processor<NeoN::Vec3>>::REGISTERED,
    &ScalarSurfBCFactory::Register<sb::Processor<NeoN::scalar>>::REGISTERED,
};
} // namespace

namespace NeoFOAM::bindings
{

void registerFieldFactories(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Field factories — read OF fields, return NeoN fields registered in db
    // -------------------------------------------------------------------
    m.def(
        "read_scalar_volume_field",
        [](nf::RunTime& rt, const std::string& name) -> fvcc::VolumeField<NeoN::scalar>&
        {
            Foam::volScalarField ofField(
                Foam::IOobject(
                    name,
                    rt.mesh.time().timeName(),
                    rt.mesh,
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh
            );
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
            return vc.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
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
        [](nf::RunTime& rt, const std::string& name) -> fvcc::VolumeField<NeoN::Vec3>&
        {
            Foam::volVectorField ofField(
                Foam::IOobject(
                    name,
                    rt.mesh.time().timeName(),
                    rt.mesh,
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh
            );
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
            return vc.registerVector<fvcc::VolumeField<NeoN::Vec3>>(
                nf::CreateFromFoamField<Foam::volVectorField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
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
        [](nf::RunTime& rt, const std::string& uFieldName) -> fvcc::SurfaceField<NeoN::scalar>&
        {
            Foam::volVectorField ofU(
                Foam::IOobject(
                    uFieldName,
                    rt.mesh.time().timeName(),
                    rt.mesh,
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh
            );
            Foam::surfaceScalarField ofPhi(
                Foam::IOobject(
                    "phi",
                    rt.mesh.time().timeName(),
                    rt.mesh,
                    Foam::IOobject::READ_IF_PRESENT,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                Foam::fvc::flux(ofU)
            );
            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
            return vc.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::surfaceScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
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
        [](nf::RunTime& rt, const std::string& name, double value)
        {
            auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
            fvcc::SurfaceField<NeoN::scalar> field(rt.exec, name, rt.nfMesh, bcs);
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
        [](nf::RunTime& rt, const std::string& fieldName, const std::string& pisoDict
        ) -> std::tuple<int, double, bool>
        {
            // Read OF field temporarily just to check needReference
            Foam::volScalarField ofField(
                Foam::IOobject(
                    fieldName,
                    rt.mesh.time().timeName(),
                    rt.mesh,
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    Foam::IOobject::NO_REGISTER
                ),
                rt.mesh
            );
            Foam::label refCell = 0;
            Foam::scalar refValue = 0.0;
            Foam::setRefCell(ofField, rt.mesh.solutionDict().subDict(pisoDict), refCell, refValue);
            bool needs = ofField.needReference() && refCell >= 0;
            return {static_cast<int>(refCell), refValue, needs};
        },
        "runtime"_a,
        "field_name"_a = std::string("p"),
        "piso_dict"_a = std::string("PISO"),
        "Get (refCell, refValue, needsReference) from fvSolution"
    );

    // -------------------------------------------------------------------
    // Uniform volume scalar field (nu, nut=0) for the explicit viscous stress.
    // Mirrors create_uniform_surface_field but on a VolumeField.
    // -------------------------------------------------------------------
    m.def(
        "create_uniform_volume_field",
        [](nf::RunTime& rt, const std::string& name, double value)
        {
            auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
            fvcc::VolumeField<NeoN::scalar> field(rt.exec, name, rt.nfMesh, bcs);
            NeoN::fill(field.internalVector(), value);
            NeoN::fill(field.boundaryData().value(), value);
            return field;
        },
        "runtime"_a,
        "name"_a,
        "value"_a,
        "Create a uniform scalar volume field (e.g. nu or nut=0)"
    );

    // Overwrite a scalar field's internal values from a host array (the closure's
    // nonlinear scalar maths — chi, fv1, fw, ... — is authored in plain NumPy).
    m.def(
        "copy_from_host",
        [](fvcc::VolumeField<NeoN::scalar>& field,
           nb::ndarray<const NeoN::scalar, nb::ndim<1>, nb::c_contig, nb::device::cpu> values)
        {
            const auto n = field.internalVector().size();
            if (static_cast<std::size_t>(values.shape(0)) != static_cast<std::size_t>(n))
            {
                throw std::runtime_error("copy_from_host: array size != field size");
            }
            field.internalVector() =
                NeoN::Vector<NeoN::scalar>(field.exec(), values.data(), n, NeoN::SerialExecutor());
            field.correctBoundaryConditions();
        },
        "field"_a,
        "values"_a,
        "Overwrite a scalar field's internal values from a host array; corrects BCs"
    );

    // NeoN binds `boundary_data_value` on its scalar VolumeField only, so the boundary
    // values a vector field's boundary conditions produced — what the patch-type
    // translation in readers.hpp is observed through — are unreachable from Python.
    m.def(
        "vector_boundary_values",
        [](fvcc::VolumeField<NeoN::Vec3>& field) -> NeoN::Vector<NeoN::Vec3>&
        { return field.boundaryData().value(); },
        "field"_a,
        nb::rv_policy::reference,
        nb::keep_alive<0, 1>(), // the returned view borrows from the field
        "Boundary values of a NeoN vector VolumeField, flat in patch order"
    );
}

} // namespace NeoFOAM::bindings
