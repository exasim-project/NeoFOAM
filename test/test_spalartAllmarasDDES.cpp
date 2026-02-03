// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <unordered_set>
#include <set>

#include "common.hpp"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"
#include "LESModel.H"

using Catch::Approx;

namespace fvc = Foam::fvc;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using Scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;
using VolScalar = fvcc::VolumeField<Scalar>;
using VolVector = fvcc::VolumeField<Vec3>;
using SurfScalar = fvcc::SurfaceField<Scalar>;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("SA-DDES: NeoN component chain + (optional) OpenFOAM nut cross-check")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    // --- NeoN database / collection
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    // Run on all executors (as in your ddtFluxCorr test)
    //auto [execName, exec] = GENERATE(allAvailableExecutor());
    std::string execName = "GPUExecutor";
    NeoN::Executor exec = NeoN::GPUExecutor {};

    // --- FOAM time
    //const Foam::scalar startTime = 0.0;
    //const Foam::label startTimeIndex = 0;
    //runTime.setTime(startTime, startTimeIndex);

    // --- Mesh + adapter runtime
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    // --- deltaT from controlDict
    const NeoN::Dictionary controlDict = nf::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");
    runTime.setDeltaT(dt);

    // --- Read FOAM fields (case must provide these)
    Foam::volVectorField U(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    );

    Foam::volScalarField p(
        Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    );

    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(U) // used only if phi file absent; MUST_READ enforces it exists.
    );
    
    // --- Viscosity from transport model (singlePhase)
    Foam::singlePhaseTransportModel transport(U, phi);
    Foam::tmp<Foam::volScalarField> tnu = transport.nu();
    Foam::volScalarField nuFoam(
        Foam::IOobject(
            "nu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tnu()
    );

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );

    const Foam::incompressible::turbulenceModel& turb = foamTurb();

    foamTurb->validate();
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");
    const Foam::volScalarField& nuTilda = mesh.lookupObject<Foam::volScalarField>("nuTilda");

    const Foam::incompressible::LESModel& lesModel =
        Foam::refCast<const Foam::incompressible::LESModel>(turb);
    const Foam::volScalarField& delta = lesModel.delta();
    Foam::wallDist y(mesh);
    const Foam::volScalarField& wallDist = y.y();
    auto tofChi = nuTilda / nuFoam;
    Foam::volScalarField ofChi(
        Foam::IOobject(
            "ofChi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofChi() // materialize tmp
    );
    auto tofChi3 = Foam::pow3(ofChi);
    Foam::volScalarField ofChi3(
        Foam::IOobject(
            "ofChi3",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofChi3()
    );
    auto tofFv1 = ofChi3 / (ofChi3 + scalar(357.911));
    Foam::volScalarField ofFv1(
        Foam::IOobject(
            "ofFv1",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofFv1()
    );

    auto tofFv2 = scalar(1) - ofChi / (scalar(1) + ofChi * ofFv1);
    Foam::volScalarField ofFv2(
        Foam::IOobject(
            "ofFv2",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofFv2()
    );

    auto tomega = Foam::sqrt(2.0) * Foam::mag(Foam::skew(fvc::grad(U)));
    Foam::volScalarField ofOmega(
        Foam::IOobject(
            "ofOmega",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tomega()
    );
    auto tmagGradU = Foam::mag(fvc::grad(U));
    Foam::volScalarField ofMagGradU(
        Foam::IOobject(
            "ofMagGradU",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tmagGradU()
    );

    scalar ofCw1 =
        scalar(0.1355) / Foam::sqr(scalar(0.41)) + (1.0 + scalar(0.622)) / scalar(0.66666);
    //ofNut = nuTilda * ofFv1;
    //ofNut.correctBoundaryConditions();

    auto tfd = 1
             - Foam::tanh(Foam::pow(
                 scalar(8)
                     * Foam::min(
                         (nuFoam + ofNut)
                             / (Foam::max(
                                    ofMagGradU,
                                    Foam::dimensionedScalar(
                                        "small",
                                        Foam::inv(Foam::dimTime),
                                        Foam::SMALL
                                    )
                                )
                                * Foam::sqr(scalar(0.41) * wallDist)),
                         scalar(10)
                     ),
                 3
             ));
    tfd.ref().boundaryFieldRef() = 0;
    Foam::volScalarField ofFd(
        Foam::IOobject(
            "ofFd",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tfd()
    );

    auto tpsi = Foam::sqrt(Foam::min(
        scalar(100),
        (1 - scalar(0.1355) / (ofCw1 * Foam::sqr(scalar(0.41)) * scalar(0.424)) * ofFv2)
            / Foam::max(Foam::SMALL, ofFv1)
    ) // ft2 = 0 (disabled)
    );
    Foam::volScalarField ofPsi(
        Foam::IOobject(
            "ofPsi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tpsi()
    );

    auto tdTilde = Foam::max(
        wallDist
            - ofFd
                  * Foam::max(
                      wallDist - ofPsi * scalar(0.65) * delta,
                      Foam::dimensionedScalar("small", Foam::dimLength, scalar(0))
                  ),
        Foam::dimensionedScalar("small", Foam::dimLength, Foam::SMALL)
    );
    Foam::volScalarField ofdTilde(
        Foam::IOobject(
            "ofdTilde",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tdTilde()
    );

    auto tsTilde = Foam::max(
        ofOmega + ofFv2 * nuTilda / Foam::sqr(scalar(0.41) * ofdTilde),
        scalar(0.3) * ofOmega
    );
    Foam::volScalarField ofsTilde(
        Foam::IOobject(
            "ofsTilde",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tsTilde()
    );
    auto tr = Foam::min(
        nuTilda
            / (Foam::max(
                   ofsTilde,
                   Foam::dimensionedScalar("small", Foam::inv(Foam::dimTime), Foam::SMALL)
               )
               * Foam::sqr(scalar(0.41) * ofdTilde)),
        scalar(10)
    );
    Foam::volScalarField r(
        Foam::IOobject(
            "r",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tr()
    );
    auto tg = r + scalar(0.3) * (Foam::pow6(r) - r);
    Foam::volScalarField g(
        Foam::IOobject(
            "g",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tg()
    );
    auto tfw = g * Foam::pow((1 + Foam::pow6(2.0)) / (Foam::pow6(g) + Foam::pow6(2.0)), 1.0 / 6.0);
    Foam::volScalarField ofFw(
        Foam::IOobject(
            "ofFw",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tfw()
    );

    auto tgradNuTilde = fvc::grad(nuTilda);
    Foam::volVectorField ofgradNutilda(
        Foam::IOobject(
            "ofgradNutilda",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tgradNuTilde()
    );
    auto tofGradU = fvc::grad(U);
    Foam::volTensorField ofGradU(
        Foam::IOobject(
            "ofgradU",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofGradU()
    );
    Foam::volVectorField ofGradUx(
        Foam::IOobject(
            "ofgradUx",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedVector("zero", ofGradU.dimensions(), Foam::vector::zero)
    );

    // fill internal field
    const Foam::label nCells = ofGradUx.internalField().size();
    for (Foam::label celli = 0; celli < nCells; ++celli)
    {
        const Foam::tensor& g = ofGradU.internalField()[celli];

        ofGradUx.internalFieldRef()[celli] = Foam::vector(
            g.xx(), // ∂Ux/∂x
            g.yx(), // ∂Ux/∂y
            g.zx()  // ∂Ux/∂z
        );
    }
    const Foam::label nPatches = ofGradUx.boundaryField().size();

    for (Foam::label patchi = 0; patchi < nPatches; ++patchi)
    {
        const Foam::fvPatchTensorField& gPatch = ofGradU.boundaryField()[patchi];

        Foam::fvPatchVectorField& uxPatch = ofGradUx.boundaryFieldRef()[patchi];

        const Foam::label nFaces = uxPatch.size();

        for (Foam::label facei = 0; facei < nFaces; ++facei)
        {
            const Foam::tensor& g = gPatch[facei];

            uxPatch[facei] = Foam::vector(
                g.xx(), // dUx/dx
                g.yx(), // dUx/dy
                g.zx()  // dUx/dz
            );
        }
    }
    Foam::volVectorField ofGradUy(
    Foam::IOobject(
        "ofgradUy",
        runTime.timeName(),
        mesh,
        Foam::IOobject::NO_READ,
        Foam::IOobject::NO_WRITE
    ),
    mesh,
    Foam::dimensionedVector("zero", ofGradU.dimensions(), Foam::vector::zero)
);

// internal field
for (Foam::label celli = 0; celli < nCells; ++celli)
{
    const Foam::tensor& g = ofGradU.internalField()[celli];

    ofGradUy.internalFieldRef()[celli] = Foam::vector(
        g.xy(), // ∂Uy/∂x
        g.yy(), // ∂Uy/∂y
        g.zy()  // ∂Uy/∂z
    );
}

// boundary field
for (Foam::label patchi = 0; patchi < nPatches; ++patchi)
{
    const Foam::fvPatchTensorField& gPatch = ofGradU.boundaryField()[patchi];
    Foam::fvPatchVectorField& uyPatch = ofGradUy.boundaryFieldRef()[patchi];

    const Foam::label nFaces = uyPatch.size();

    for (Foam::label facei = 0; facei < nFaces; ++facei)
    {
        const Foam::tensor& g = gPatch[facei];

        uyPatch[facei] = Foam::vector(
            g.xy(), // dUy/dx
            g.yy(), // dUy/dy
            g.zy()  // dUy/dz
        );
    }
}
Foam::volVectorField ofGradUz(
    Foam::IOobject(
        "ofgradUz",
        runTime.timeName(),
        mesh,
        Foam::IOobject::NO_READ,
        Foam::IOobject::NO_WRITE
    ),
    mesh,
    Foam::dimensionedVector("zero", ofGradU.dimensions(), Foam::vector::zero)
);

// internal field
for (Foam::label celli = 0; celli < nCells; ++celli)
{
    const Foam::tensor& g = ofGradU.internalField()[celli];

    ofGradUz.internalFieldRef()[celli] = Foam::vector(
        g.xz(), // ∂Uz/∂x
        g.yz(), // ∂Uz/∂y
        g.zz()  // ∂Uz/∂z
    );
}

// boundary field
for (Foam::label patchi = 0; patchi < nPatches; ++patchi)
{
    const Foam::fvPatchTensorField& gPatch = ofGradU.boundaryField()[patchi];
    Foam::fvPatchVectorField& uzPatch = ofGradUz.boundaryFieldRef()[patchi];

    const Foam::label nFaces = uzPatch.size();

    for (Foam::label facei = 0; facei < nFaces; ++facei)
    {
        const Foam::tensor& g = gPatch[facei];

        uzPatch[facei] = Foam::vector(
            g.xz(), // dUz/dx
            g.yz(), // dUz/dy
            g.zz()  // dUz/dz
        );
    }
}

    auto tmagSqrGradNuTilde = Foam::magSqr(ofgradNutilda);
    Foam::volScalarField ofmagSqrGradNutilda(
        Foam::IOobject(
            "ofmagSqrGradNutilda",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tmagSqrGradNuTilde()
    );
    auto tproduction =
        scalar(0.1355) * ofsTilde * nuTilda + scalar(0.622 / 0.66666) * ofmagSqrGradNutilda;
    Foam::volScalarField ofProduction(
        Foam::IOobject(
            "ofProduction",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tproduction()
    );
    auto tspCoeff = ofCw1 * ofFw * nuTilda / Foam::sqr(ofdTilde);
    Foam::volScalarField ofspCoeff(
        Foam::IOobject(
            "ofspCoeff",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tspCoeff()
    );
    auto tviscousStress = fvc::laplacian(ofNut, U, "laplacian(nuEff,U)")
                        - fvc::div(nuFoam * Foam::dev2(Foam::T(fvc::grad(U))));
    Foam::volVectorField ofViscousStress(
        Foam::IOobject(
            "ofViscousStress",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tviscousStress()
    );

    auto tofTau = Foam::dev2(Foam::T(fvc::grad(U)));
    Foam::volTensorField ofTau(
        Foam::IOobject(
            "ofTau",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofTau()
    );
    const Foam::tensor& t0 = ofTau.internalField()[0];

    Foam::Info
        << "ofTau[0] = " << t0 << Foam::endl
        << "  xx=" << t0.xx()
        << " xy=" << t0.xy()
        << " xz=" << t0.xz() << Foam::endl
        << "  yx=" << t0.yx()
        << " yy=" << t0.yy()
        << " yz=" << t0.yz() << Foam::endl
        << "  zx=" << t0.zx()
        << " zy=" << t0.zy()
        << " zz=" << t0.zz()
        << Foam::endl;
    Foam::volVectorField ofTaux(
        Foam::IOobject(
            "ofTaux",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedVector("zero", ofGradU.dimensions(), Foam::vector::zero)
    );

    // fill internal field
    //const Foam::label nCells = ofTaux.internalField().size();
    for (Foam::label celli = 0; celli < nCells; ++celli)
    {
        const Foam::tensor& g = ofTau.internalField()[celli];

        ofTaux.internalFieldRef()[celli] = Foam::vector(
            g.xx(), // ∂Ux/∂x
            g.yx(), // ∂Ux/∂y
            g.zx()  // ∂Ux/∂z
        );
    }
    //const Foam::label nPatches = ofTaux.boundaryField().size();

    for (Foam::label patchi = 0; patchi < nPatches; ++patchi)
    {
        const Foam::fvPatchTensorField& gPatch = ofTau.boundaryField()[patchi];

        Foam::fvPatchVectorField& uxPatch = ofTaux.boundaryFieldRef()[patchi];

        const Foam::label nFaces = uxPatch.size();

        for (Foam::label facei = 0; facei < nFaces; ++facei)
        {
            const Foam::tensor& g = gPatch[facei];

            uxPatch[facei] = Foam::vector(
                g.xx(), // dUx/dx
                g.yx(), // dUx/dy
                g.zx()  // dUx/dz
            );
        }
    }

    // === Mirror state into NeoN ===
    auto& nfU =
        fieldCollection.registerVector<VolVector>(nf::CreateFromFoamField<Foam::volVectorField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = U,
            .name = "nfU"
        });

    auto& nfp =
        fieldCollection.registerVector<VolScalar>(nf::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = p,
            .name = "nfp"
        });

    auto& nfPhi = fieldCollection.registerVector<SurfScalar>(
        nf::CreateFromFoamField<Foam::surfaceScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = phi,
            .name = "nfPhi"
        }
    );

    auto& nfWallDist =
        fieldCollection.registerVector<VolScalar>(nf::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = wallDist,
            .name = "nfWallDist"
        });

    auto& nfNuTilda =
        fieldCollection.registerVector<VolScalar>(nf::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = nuTilda,
            .name = "nfNuTilda"
        });

    auto& nut =
        fieldCollection.registerVector<VolScalar>(nf::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = ofNut,
            .name = "nut"
        });

    auto& nfDelta =
        fieldCollection.registerVector<VolScalar>(nf::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = delta,
            .name = "nfDelta"
        });

    // --- Constant nu field in NeoN
    auto volCalcBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Scalar>>(rt.nfMesh);
    auto volCalcVecBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(rt.nfMesh);
    auto surfCalcBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Scalar>>(rt.nfMesh);

    Foam::IOdictionary transportProperties(Foam::IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        Foam::IOobject::MUST_READ_IF_MODIFIED,
        Foam::IOobject::NO_WRITE
    ));

    Foam::dimensionedScalar viscosity("nu", Foam::dimViscosity, transportProperties);
    VolScalar nfNu(exec, "nu", rt.nfMesh, volCalcBCs);
    NeoN::fill(nfNu.internalVector(), viscosity.value());
    NeoN::fill(nfNu.boundaryData().value(), viscosity.value());

    NeoN::turbulenceModels::SpalartAllmarasDDES saBase(rt.exec, rt.nfMesh);

    auto gradOp = nnfvcc::GaussGreenGrad(exec, rt.nfMesh);
    fvcc::GradVecField G{
        VolVector(exec, "gradUx", rt.nfMesh, volCalcVecBCs),
        VolVector(exec, "gradUy", rt.nfMesh, volCalcVecBCs),
        VolVector(exec, "gradUz", rt.nfMesh, volCalcVecBCs)
    }; 
    gradOp.grad(nfU,G);
  //  nf::compare(G.gradUx, ofGradUx, ApproxVector(1e-10), true);
  //  nf::compare(G.gradUy, ofGradUy, ApproxVector(1e-10), true);
  //  nf::compare(G.gradUz, ofGradUz, ApproxVector(1e-10), true);

    VolVector gradNuTilda(exec, "gradNuTilda", rt.nfMesh, volCalcVecBCs);
    VolScalar magSqrGradNuTilda(exec, "magSqrGradNuTilda", rt.nfMesh, volCalcBCs);
    SurfScalar nfSurfNuEff(exec, "nfSurfNuEff", rt.nfMesh, surfCalcBCs);
    SurfScalar nfSurfNu(exec, "nfSurfNu", rt.nfMesh, surfCalcBCs);
    SurfScalar nfSurfNut(exec, "nfSurfNut", rt.nfMesh, surfCalcBCs);
    SurfScalar nfSurfNuTilda(exec, "nfSurfNuTilda", rt.nfMesh, surfCalcBCs);
    SurfScalar nuTildaEff(exec, "nuTildaEff", rt.nfMesh, surfCalcBCs);
    VolScalar nfFusedProduction(exec, "nfFusedProduction", rt.nfMesh, volCalcBCs);
    VolScalar nfFusedSpCoeff(exec, "nfFusedSpCoeff", rt.nfMesh, volCalcBCs);

    const auto& coeffs = saBase.coeffs();
    const auto Cw1 = saBase.cw1();
    const auto sigmaNut = coeffs.sigmaNut;

   // REQUIRE(Cw1 == ofCw1);

    auto surfInterpol = fvcc::SurfaceInterpolation<Scalar>(
        rt.exec,
        rt.nfMesh,
        NeoN::TokenList({std::string("linear")})
    );
    surfInterpol.interpolate(nfNu,nfSurfNu);

    saBase.correctNut(nut,nfSurfNut,nfSurfNuEff, nfNuTilda, nfNu,nfSurfNu);
//    nf::compare(nut, ofNut, ApproxScalar(1e-12), true);
//    nf::compare(nfNuTilda, nuTilda, ApproxScalar(1e-12));
    saBase.calcNuTildaDiffusionCoeff(nfNuTilda,nfSurfNu,nfSurfNuTilda, nuTildaEff);

    fvcc::rotateOldTimes(nfU);
    fvcc::rotateOldTimes(nfPhi);
    //fvcc::rotateOldTimes(nfp);
    fvcc::rotateOldTimes(nfNuTilda);
    gradOp.grad(nfNuTilda,gradNuTilda);
  //  nf::compare(gradNuTilda, ofgradNutilda, ApproxVector(1e-12), true);
    saBase.calcMagSqrVec(magSqrGradNuTilda, gradNuTilda);
  //  nf::compare(magSqrGradNuTilda, ofmagSqrGradNutilda, ApproxScalar(1e-12), false);

    saBase.computeProdSpDDES(
        nfFusedProduction,
        nfFusedSpCoeff,
        nfNuTilda,
        nfNu,
	G.gradUx,
	G.gradUy,
	G.gradUz,
        nfWallDist,
        nfDelta,
        magSqrGradNuTilda
    );
  //  nf::compare(nfFusedProduction, ofProduction, ApproxScalar(1e-12), false);
 //   nf::compare(nfFusedSpCoeff, ofspCoeff, ApproxScalar(1e-12), false);

    NeoN::TokenList interpolationScheme;
    interpolationScheme.insert(std::string("linear"));
    interpolationScheme.insert(std::string("uncorrected"));
    fvcc::GaussViscousStress opVisc(exec, rt.nfMesh, interpolationScheme);
    auto nfViscousStress = opVisc.viscousStress(
		    nfSurfNu, nfSurfNut, nfSurfNuTilda, nfU, G.gradUx, G.gradUy, G.gradUz, dsl::Coeff(1.0));
 //   nf::compare(nfViscousStress, ofViscousStress, ApproxVector(1e-12), false);
/*    auto hTau = nfViscousStress.internalVector().copyToHost();
    auto hView = hTau.view({0, 1});
    const NeoN::Vec3& taux0 = hView[0];

    NeoN::Logging::info(
        "nf tauX[0] = ({}, {}, {})",
        taux0[0], taux0[1], taux0[2]
    );  */
//    nf::compare(nfViscousStress, ofTaux, ApproxVector(1e-10), false);

    //nf::compare(nfU, U, ApproxVector(1e-12));
    nf::PDESolver<NeoN::Vec3> UEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfSurfNuEff, nfU)
            + dsl::exp::viscousStress(nfSurfNu, nfSurfNut, nfSurfNuTilda, nfU, G.gradUx, G.gradUy, G.gradUz),
        nfU,
        rt
    );
    UEqn.solve(-1.0 * dsl::exp::grad(nfp));
    nfU.correctBoundaryConditions();
    
    nf::PDESolver<NeoN::scalar> nuTildaEqn(
        dsl::imp::ddt(nfNuTilda) + dsl::imp::div(nfPhi, nfNuTilda)
            - dsl::imp::laplacian(nuTildaEff, nfNuTilda)
            + dsl::imp::source(nfFusedSpCoeff, nfNuTilda)
            - dsl::exp::sourceU(nfFusedProduction),
        nfNuTilda,
        rt
    );
    nuTildaEqn.solve();
    //nut.correctBoundaryConditions(nfU,nfNu);
    saBase.calcNuTildaDiffusionCoeff(nfNuTilda,nfSurfNu,nfSurfNuTilda, nuTildaEff);
    saBase.correctNut(nut,nfSurfNut,nfSurfNuEff, nfNuTilda, nfNu,nfSurfNu);
    // -----------------------------
    // OpenFOAM nut cross-check
    // -----------------------------
    SECTION("OpenFOAM incompressible::turbulenceModel nut matches (" + execName + ")")
    {
	
	 Foam::fvVectorMatrix ofUEqn
         (
             Foam::fvm::ddt(U) + Foam::fvm::div(phi, U)
         //    + foamTurb->divDevReff(U)
	     - Foam::fvm::laplacian(ofNut+nuFoam, U)
	     + fvc::laplacian(ofNut, U, "laplacian(nuEff,U)")
             - fvc::div(nuFoam * Foam::dev2(Foam::T(fvc::grad(U))))
	//     + ofViscousStress
        //     ==
        //     Foam::fvOptions(U)
        );
	Foam::solve(ofUEqn == -fvc::grad(p));
	U.correctBoundaryConditions();
    //    nf::compare(nfU, U, ApproxVector(1e-9));

	Foam::Info << "turbulence model: " << foamTurb->type() << Foam::endl;
	Foam::Info << "max|R_| = " << Foam::gMax(Foam::mag(foamTurb->R())().primitiveField())
        << Foam::endl;
        // Build OpenFOAM turbulence model from dictionaries.
        foamTurb->correct();

   	const Foam::volScalarField& nutFoam = mesh.lookupObject<Foam::volScalarField>("nut");
   	const Foam::volScalarField& nuTildaFoam = mesh.lookupObject<Foam::volScalarField>("nuTilda");
	Foam::surfaceScalarField nuEffFoam(
            Foam::IOobject(
                "nuEffFoam",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            fvc::interpolate(nutFoam+nuFoam)
        );
   //     nf::compare(nfNuTilda, nuTildaFoam, ApproxScalar(1e-12));
   //     nf::compare(nut, nutFoam, ApproxScalar(1e-12));
   //     nf::compare(nfSurfNuEff, nuEffFoam, ApproxScalar(1e-12));
    }  
}
