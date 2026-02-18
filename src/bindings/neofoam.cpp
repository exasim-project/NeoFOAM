// SPDX-FileCopyrightText: 2024-2026 NeoFOAM authors
// SPDX-License-Identifier: Unlicense

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/algorithms/pressureVelocityCoupling.hpp"
#include "NeoFOAM/datastructures/pdeSolver.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/compatibility/fvSchemes.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/auxiliary/writers.hpp"
#include "NeoFOAM/auxiliary/setup.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

// OpenFOAM headers
#include "fvCFD.H"
#include "pisoControl.H"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;
namespace dsl = NeoN::dsl;

// ===========================================================================
// Building block 1: Runtime — OpenFOAM Time + NeoFOAM RunTime adapter
// ===========================================================================
class Runtime
{
    std::vector<std::string> argStrings_;
    std::vector<char*> cArgs_;
    char** argv_ = nullptr; // lvalue needed by Foam::argList(int&, char**&)
    int argc_ = 0;

    std::unique_ptr<Foam::argList> args_;
    Foam::Time* foamTime_ = nullptr;
    std::unique_ptr<nf::RunTime> rt_;

public:

    explicit Runtime(std::vector<std::string> argv)
    {
        argStrings_ = std::move(argv);
        argc_ = static_cast<int>(argStrings_.size());
        cArgs_.resize(argc_ + 1, nullptr);
        for (int i = 0; i < argc_; ++i)
            cArgs_[i] = const_cast<char*>(argStrings_[i].c_str());
        cArgs_[argc_] = nullptr;
        argv_ = cArgs_.data();

        args_ = std::make_unique<Foam::argList>(argc_, argv_);
        foamTime_ = new Foam::Time(Foam::Time::controlDictName, *args_);
        rt_ = std::make_unique<nf::RunTime>(nf::createAdapterRunTime(*foamTime_));

        // Map fvSolution solver entries for Ginkgo
        auto& solverDict = rt_->fvSolutionDict.subDict("solvers");
        for (auto& name : solverDict.keys())
        {
            solverDict.subDict(name) = nf::mapFvSolution(solverDict.subDict(name));
        }

        // Map fvSchemes (mirrors C++ neoIcoFoam.cpp line: schemesDict = nf::mapFvSchemes(...))
        rt_->fvSchemesDict = nf::mapFvSchemes(rt_->fvSchemesDict);
    }

    ~Runtime()
    {
        rt_.reset();
        delete foamTime_;
        foamTime_ = nullptr;
        args_.reset();
    }

    // Time loop
    bool loop() { return foamTime_->loop(); }
    double time() const { return foamTime_->time().value(); }
    double deltaT() const { return foamTime_->deltaTValue(); }
    std::string timeName() const { return std::string(foamTime_->timeName()); }

    // Sync NeoN runtime from OF (time, deltaT, adjustable timestep)
    void sync(double coNum) { nf::syncRunTimes(*foamTime_, *rt_, coNum); }

    // IO
    void write() { foamTime_->write(); }
    bool outputTime() { return foamTime_->outputTime(); }
    void printExecutionTime() { foamTime_->printExecutionTime(Foam::Info); }

    // Access NeoFOAM RunTime (for PDESolver, field registration, ...)
    nf::RunTime& nfRuntime() { return *rt_; }
    const nf::RunTime& nfRuntime() const { return *rt_; }

    // Convenience accessors forwarding into nf::RunTime
    const NeoN::Executor& executor() const { return rt_->exec; }
    NeoN::UnstructuredMesh& nfMesh() { return rt_->nfMesh; }
    const NeoN::UnstructuredMesh& nfMesh() const { return rt_->nfMesh; }
    NeoN::Database& db() { return rt_->db; }
    nf::MeshAdapter& mesh() { return rt_->mesh; }
    const nf::MeshAdapter& mesh() const { return rt_->mesh; }

    // Low-level OF Time (needed by PisoControl, field readers)
    Foam::Time& foamTime() { return *foamTime_; }
};

// ===========================================================================
// Building block 2: PisoControl — wraps Foam::pisoControl
// ===========================================================================
class PisoControl
{
    std::unique_ptr<Foam::pisoControl> piso_;

public:

    explicit PisoControl(Runtime& rt)
        : piso_(std::make_unique<Foam::pisoControl>(rt.mesh()))
    {}

    bool momentumPredictor() { return piso_->momentumPredictor(); }
    bool correct() { return piso_->correct(); }
    bool correctNonOrthogonal() { return piso_->correctNonOrthogonal(); }
    bool finalNonOrthogonalIter() { return piso_->finalNonOrthogonalIter(); }
};

// ===========================================================================
// Module definition
// ===========================================================================
NB_MODULE(neofoam_bindings, m)
{
    m.doc() = "NeoFOAM Python bindings — building blocks for NeoN-based solvers";

    // -------------------------------------------------------------------
    // Runtime
    // -------------------------------------------------------------------
    nb::class_<Runtime>(m, "Runtime")
        .def(
            nb::init<std::vector<std::string>>(),
            "argv"_a,
            "Create runtime from command-line arguments (creates OF Time + NeoN mesh)"
        )
        // Time loop
        .def("loop", &Runtime::loop, "Advance time loop; False when finished")
        .def("time", &Runtime::time)
        .def("delta_t", &Runtime::deltaT)
        .def("time_name", &Runtime::timeName)
        .def("sync", &Runtime::sync, "co_num"_a, "Sync OF ↔ NeoN time/deltaT")
        // IO
        .def("write", &Runtime::write)
        .def("output_time", &Runtime::outputTime)
        .def("print_execution_time", &Runtime::printExecutionTime)
        // NeoFOAM RunTime
        .def(
            "nf_runtime",
            [](Runtime& self) -> nf::RunTime& { return self.nfRuntime(); },
            nb::rv_policy::reference_internal
        )
        .def("executor", [](Runtime& self) -> NeoN::Executor { return self.executor(); })
        .def(
            "nf_mesh",
            [](Runtime& self) -> const NeoN::UnstructuredMesh& { return self.nfMesh(); },
            nb::rv_policy::reference_internal
        )
        .def(
            "db",
            [](Runtime& self) -> NeoN::Database& { return self.db(); },
            nb::rv_policy::reference_internal
        );

    // -------------------------------------------------------------------
    // PisoControl
    // -------------------------------------------------------------------
    nb::class_<PisoControl>(m, "PisoControl")
        .def(nb::init<Runtime&>(), "runtime"_a, nb::keep_alive<1, 2>())
        .def("momentum_predictor", &PisoControl::momentumPredictor)
        .def("correct", &PisoControl::correct)
        .def("correct_non_orthogonal", &PisoControl::correctNonOrthogonal)
        .def("final_non_orthogonal_iter", &PisoControl::finalNonOrthogonalIter);

    // -------------------------------------------------------------------
    // RunTime (read-only NeoFOAM struct, needed by PDESolver)
    // -------------------------------------------------------------------
    nb::class_<nf::RunTime>(m, "RunTime")
        .def_ro("t", &nf::RunTime::t)
        .def_ro("dt", &nf::RunTime::dt);

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

    // -------------------------------------------------------------------
    // Field writers — write NeoN fields back through OpenFOAM IO
    // -------------------------------------------------------------------
    m.def(
        "write_scalar_field",
        [](const fvcc::VolumeField<NeoN::scalar>& field, Runtime& rt)
        { nf::write(field, rt.mesh()); },
        "field"_a,
        "runtime"_a,
        "Write a NeoN scalar VolumeField via OpenFOAM IO"
    );

    m.def(
        "write_vector_field",
        [](const fvcc::VolumeField<NeoN::Vec3>& field, Runtime& rt)
        { nf::write(field, rt.mesh()); },
        "field"_a,
        "runtime"_a,
        "Write a NeoN vector VolumeField via OpenFOAM IO"
    );

    // -------------------------------------------------------------------
    // DdtScheme enum (NeoN)
    // -------------------------------------------------------------------
    nb::enum_<fvcc::DdtScheme>(m, "DdtScheme")
        .value("None", fvcc::DdtScheme::None)
        .value("BDF1", fvcc::DdtScheme::BDF1)
        .value("BDF2", fvcc::DdtScheme::BDF2);

    // -------------------------------------------------------------------
    // PDESolver<scalar>
    // -------------------------------------------------------------------
    nb::class_<nf::PDESolver<NeoN::scalar>>(m, "PDESolverScalar")
        .def(
            "__init__",
            [](nf::PDESolver<NeoN::scalar>& self,
               dsl::Expression<NeoN::scalar> expr,
               fvcc::VolumeField<NeoN::scalar>& psi,
               const nf::RunTime& rt)
            { new (&self) nf::PDESolver<NeoN::scalar>(std::move(expr), psi, rt); },
            "expr"_a,
            "psi"_a,
            "runtime"_a,
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>()
        )
        .def(
            "solve",
            [](nf::PDESolver<NeoN::scalar>& self) { return self.solve(); },
            "Solve the linear system"
        )
        .def(
            "assemble",
            [](nf::PDESolver<NeoN::scalar>& self) -> void { self.assemble(); },
            "Assemble the linear system"
        )
        .def(
            "set_reference",
            &nf::PDESolver<NeoN::scalar>::setReference,
            "ref_cell"_a,
            "ref_value"_a,
            "Set pressure reference cell and value"
        );

    // -------------------------------------------------------------------
    // PDESolver<Vec3>
    // -------------------------------------------------------------------
    nb::class_<nf::PDESolver<NeoN::Vec3>>(m, "PDESolverVec3")
        .def(
            "__init__",
            [](nf::PDESolver<NeoN::Vec3>& self,
               dsl::Expression<NeoN::Vec3> expr,
               fvcc::VolumeField<NeoN::Vec3>& psi,
               const nf::RunTime& rt)
            { new (&self) nf::PDESolver<NeoN::Vec3>(std::move(expr), psi, rt); },
            "expr"_a,
            "psi"_a,
            "runtime"_a,
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>()
        )
        .def(
            "solve",
            [](nf::PDESolver<NeoN::Vec3>& self) { return self.solve(); },
            "Solve the linear system"
        )
        .def(
            "solve_with_source",
            [](nf::PDESolver<NeoN::Vec3>& self, dsl::SpatialOperator<NeoN::Vec3> rhs)
            { return self.solve(std::move(rhs)); },
            "rhs"_a,
            "Solve with an explicit source term (e.g. -grad(p))"
        )
        .def(
            "assemble",
            [](nf::PDESolver<NeoN::Vec3>& self) -> void { self.assemble(); },
            "Assemble the linear system"
        )
        .def(
            "ddt_scheme",
            &nf::PDESolver<NeoN::Vec3>::ddtScheme,
            "Get the ddt scheme determined from fvSchemes"
        );

    // -------------------------------------------------------------------
    // Pressure-velocity coupling helpers (free functions)
    // -------------------------------------------------------------------
    m.def(
        "compute_rau_and_hbya",
        [](const nf::PDESolver<NeoN::Vec3>& UEqn) { return nf::computeRAUandHByA(UEqn); },
        "UEqn"_a,
        "Compute rAU and HbyA from the assembled momentum equation"
    );

    m.def(
        "constrain_hbya",
        [](const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::VolumeField<NeoN::scalar>& p,
           fvcc::VolumeField<NeoN::Vec3>& hByA) { nf::constrainHbyA(U, p, hByA); },
        "U"_a,
        "p"_a,
        "hByA"_a,
        "Constrain HbyA at boundaries with assigned velocity BCs"
    );

    m.def(
        "flux",
        [](const fvcc::VolumeField<NeoN::Vec3>& volField) { return nf::flux(volField); },
        "vol_field"_a,
        "Compute face flux from a volume vector field"
    );

    m.def(
        "update_face_velocity",
        [](const fvcc::SurfaceField<NeoN::scalar>& phiHbyA,
           const nf::PDESolver<NeoN::scalar>& pEqn,
           fvcc::SurfaceField<NeoN::scalar>& phi) { nf::updateFaceVelocity(phiHbyA, pEqn, phi); },
        "phi_hbya"_a,
        "pEqn"_a,
        "phi"_a,
        "Update face velocity (phi) after pressure correction"
    );

    m.def(
        "update_velocity",
        [](const fvcc::VolumeField<NeoN::Vec3>& hByA,
           const fvcc::VolumeField<NeoN::scalar>& rAU,
           const fvcc::VolumeField<NeoN::scalar>& p,
           fvcc::VolumeField<NeoN::Vec3>& U) { nf::updateVelocity(hByA, rAU, p, U); },
        "hByA"_a,
        "rAU"_a,
        "p"_a,
        "U"_a,
        "Update cell velocity: U = HbyA - rAU * grad(p)"
    );

    m.def(
        "ddt_flux_corr",
        [](const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double dt,
           fvcc::DdtScheme scheme)
        { return fvcc::ddtFluxCorr(U, phi, static_cast<NeoN::scalar>(dt), scheme); },
        "U"_a,
        "phi"_a,
        "dt"_a,
        "scheme"_a,
        "Compute ddt flux correction for PISO loop (BDF1/BDF2)"
    );

    // -------------------------------------------------------------------
    // Utility: read a dimensioned scalar from OpenFOAM constant/ dict
    // -------------------------------------------------------------------
    m.def(
        "read_transport_viscosity",
        [](Runtime& rt) -> double
        {
            Foam::IOdictionary transportProperties(Foam::IOobject(
                "transportProperties",
                rt.foamTime().constant(),
                rt.mesh(),
                Foam::IOobject::MUST_READ_IF_MODIFIED,
                Foam::IOobject::NO_WRITE
            ));
            Foam::dimensionedScalar nu("nu", Foam::dimViscosity, transportProperties);
            return nu.value();
        },
        "runtime"_a,
        "Read kinematic viscosity nu from constant/transportProperties"
    );

    // -------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------
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
