// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"  // <- your FoamAdapter test helpers (mesh/time adapters, converters, compare, etc.)
#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

using Catch::Approx;

namespace fvc  = Foam::fvc;
namespace fvcc = NeoN::finiteVolume::cellCentred;

using Scalar     = NeoN::scalar;
using Vec3       = NeoN::Vec3;
using VolScalar  = fvcc::VolumeField<Scalar>;
using VolVector  = fvcc::VolumeField<Vec3>;
using SurfScalar = fvcc::SurfaceField<Scalar>;

// For MSVC
template class NeoN::timeIntegration::ForwardEuler<VolScalar>;

extern Foam::Time* timePtr; // provided by the test harness

TEST_CASE("ddtCorr: OpenFOAM Euler vs NeoN (Forward Euler)")
{
    Foam::Time& runTime = *timePtr;

    // Create the NeoN db/collection
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    //auto [execName, exec] = GENERATE(allAvailableExecutor());
    const Foam::word execName = "GPU";
    auto exec = NeoN::GPUExecutor();

    // Prepare FOAM time
    const Foam::scalar startTime = 0.0;
    const Foam::label startTimeIndex = 0;
    runTime.setTime(startTime, startTimeIndex);

    // Mesh adapters
    std::unique_ptr<NeoFOAM::MeshAdapter> meshAdapterPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshAdapterPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    // Time-step (use controlDict deltaT to keep both sides identical)
    const NeoN::Dictionary controlDict = NeoFOAM::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");
    // Make sure we advance once so oldTime slots are established in FOAM
    runTime.setDeltaT(dt);

    // FOAM fields (vector U, surface-scalar phi)
    Foam::volVectorField U
    (
        Foam::IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    );

    Foam::surfaceScalarField phi
    (
        Foam::IOobject
        (
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
	fvc::flux(U)
    );

    // === OpenFOAM: compute ddtPhiCorr(U, phi) for Euler ===
    Foam::surfaceScalarField foamCorr = fvc::ddtCorr(U, phi);

    // === NeoN: mirror the same state and compute forwardEuler::ddtPhiCorr ===

    // Register U (VolVector) 
    auto& nfU = fieldCollection.registerVector<fvcc::VolumeField<Vec3>>(
        NeoFOAM::CreateFromFoamField<Foam::volVectorField>
        {
            .exec     = exec,
            .nfMesh   = nfMesh,
            .foamField= U,
            .name     = "nfU"
        }
    );

    // Register phi  
    auto& nfPhi = fieldCollection.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
        NeoFOAM::CreateFromFoamField<Foam::surfaceScalarField>{
            .exec = exec,
            .nfMesh = nfMesh,
            .foamField = phi,
            .name = "nfPhi"
        }
    );

    auto& nfU0   = fvcc::oldTime(nfU);
    auto& nfPhi0 = fvcc::oldTime(nfPhi);

    // Time integrator
    NeoN::Dictionary fvSchemes, ddtSchemes, fvSolution;
    ddtSchemes.insert("type", std::string("forwardEuler"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoN::timeIntegration::ForwardEuler<VolScalar> fe(fvSchemes.subDict("ddtSchemes"), fvSolution);

    // Compute correction
    SurfScalar nfCorr = fe.ddtPhiCorr(nfU, nfPhi, dt);

    // === Compare foamCorr vs nfCorr ===
    auto nfCorrHost   = nfCorr.internalVector().copyToHost();

    NeoFOAM::compare(nfPhi, phi, ApproxScalar(1e-15));
    NeoFOAM::compare(nfPhi0, phi.oldTime(), ApproxScalar(1e-15));
    NeoFOAM::compare(nfU, U, ApproxVector(1e-15));
    SECTION("ddtCorr " + execName)
    {
        NeoFOAM::compare(nfCorr, foamCorr, ApproxScalar(1e-15));
    }
}

