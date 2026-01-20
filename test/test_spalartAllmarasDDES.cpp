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
namespace nf   = NeoFOAM;

using Scalar     = NeoN::scalar;
using Vec3       = NeoN::Vec3;
using VolScalar  = fvcc::VolumeField<Scalar>;
using VolVector  = fvcc::VolumeField<Vec3>;
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
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // --- FOAM time
    const Foam::scalar startTime = 0.0;
    const Foam::label  startTimeIndex = 0;
    runTime.setTime(startTime, startTimeIndex);

    // --- Mesh + adapter runtime
    auto rt   = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    // --- deltaT from controlDict
    const NeoN::Dictionary controlDict = nf::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");
    runTime.setDeltaT(dt);

    // --- Read FOAM fields (case must provide these)
    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        mesh
    );

    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        mesh
    );

    Foam::surfaceScalarField phi(
        Foam::IOobject("phi", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        fvc::flux(U) // used only if phi file absent; MUST_READ enforces it exists.
    );

    Foam::volScalarField nuTilda(
        Foam::IOobject("nuTilda", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        mesh
    );
    nuTilda.correctBoundaryConditions();
    Foam::volScalarField nut(
        Foam::IOobject("nut", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        mesh
    );

    // --- Viscosity from transport model (singlePhase)
    Foam::singlePhaseTransportModel transport(U, phi);
    Foam::tmp<Foam::volScalarField> tnu = transport.nu();
    Foam::volScalarField nuFoam(
        Foam::IOobject("nu", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tnu()
    );

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb
    (
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );

    const Foam::incompressible::turbulenceModel& turb = foamTurb();

    const Foam::incompressible::LESModel& lesModel =
    Foam::refCast<const Foam::incompressible::LESModel>(turb);
    const Foam::volScalarField& delta = lesModel.delta();
    Foam::wallDist y(mesh);
    const Foam::volScalarField& wallDist = y.y();
    auto tofChi = nuTilda / nuFoam;
    Foam::volScalarField ofChi
    (
        Foam::IOobject
        (
            "ofChi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tofChi()   // materialize tmp
    );
    auto tofChi3 = Foam::pow3(ofChi);
    Foam::volScalarField ofChi3
    (
        Foam::IOobject("ofChi3", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tofChi3()
    ); 
    auto tofFv1 = ofChi3 / ( ofChi3 + scalar(357.911));
    Foam::volScalarField ofFv1
    (
        Foam::IOobject("ofFv1", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tofFv1()
    );
    auto tofFv2 = scalar(1) - ofChi/(scalar(1) + ofChi*ofFv1);
    Foam::volScalarField ofFv2
    (
        Foam::IOobject("ofFv2", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tofFv2()
    );

    auto tomega = Foam::sqrt(2.0)*Foam::mag(Foam::skew(fvc::grad(U)));
    Foam::volScalarField ofOmega
    (
        Foam::IOobject("ofOmega", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tomega()
    );
    auto tmagGradU = Foam::mag(fvc::grad(U));
    Foam::volScalarField ofMagGradU
    (
        Foam::IOobject("ofMagGradU", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tmagGradU()
    );

    scalar ofCw1 = scalar(0.1355)/Foam::sqr(scalar(0.41)) + (1.0 + scalar(0.622))/scalar(0.66666);
    nut = nuTilda * ofFv1;
    nut.correctBoundaryConditions();
    auto tfd = 1 - Foam::tanh(Foam::pow(scalar(8)*Foam::min((nuFoam+nut)/(Foam::max(ofMagGradU, Foam::dimensionedScalar("small", Foam::inv(Foam::dimTime), Foam::SMALL))*Foam::sqr(scalar(0.41)*wallDist)), scalar(10)),3));
    tfd.ref().boundaryFieldRef() = 0;
    Foam::volScalarField ofFd
    (
        Foam::IOobject("ofFd", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tfd()
    );

    auto tpsi = Foam::sqrt
            (
                Foam::min
                (
                    scalar(100),
                    (1 - scalar(0.1355)/(ofCw1*Foam::sqr(scalar(0.41))*scalar(0.424))
                   *ofFv2)
                   /Foam::max(Foam::SMALL, ofFv1))    // ft2 = 0 (disabled)
                );
    Foam::volScalarField ofPsi
    (
        Foam::IOobject("ofPsi", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tpsi()
    );

    auto tdTilda = Foam::max(wallDist - ofFd*Foam::max(wallDist - ofPsi*scalar(0.65)*delta,Foam::dimensionedScalar("small", Foam::dimLength, scalar(0))),Foam::dimensionedScalar("small", Foam::dimLength, Foam::SMALL));
    Foam::volScalarField ofdTilda
    (
        Foam::IOobject("ofdTilda", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tdTilda()
    );

    auto tsTilda = Foam::max(ofOmega + ofFv2*nuTilda/Foam::sqr(scalar(0.41)*ofdTilda),scalar(0.3)*ofOmega);
    Foam::volScalarField ofsTilda
    (
        Foam::IOobject("ofsTilda", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tsTilda()
    );
    auto tr = Foam::min(nuTilda/(Foam::max(ofsTilda, Foam::dimensionedScalar("small", Foam::inv(Foam::dimTime), Foam::SMALL))*Foam::sqr(scalar(0.41)*ofdTilda)), scalar(10));
    Foam::volScalarField r
    (
        Foam::IOobject("r", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tr()
    );
    auto tg = r + scalar(0.3)*(Foam::pow6(r) - r);
    Foam::volScalarField g
    (
        Foam::IOobject("g", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tg()
    );
    auto tfw = g*Foam::pow((1 + Foam::pow6(2.0))/(Foam::pow6(g) + Foam::pow6(2.0)), 1.0/6.0);
    Foam::volScalarField ofFw
    (
        Foam::IOobject("ofFw", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tfw()
    );
    auto tgradNuTilda = fvc::grad(nuTilda);
    Foam::volVectorField ofgradNutilda
    (
        Foam::IOobject("ofgradNutilda", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tgradNuTilda()
    );
    auto tmagSqrGradNuTilda = Foam::magSqr(ofgradNutilda);
    Foam::volScalarField ofmagSqrGradNutilda
    (
        Foam::IOobject("ofmagSqrGradNutilda", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tmagSqrGradNuTilda()
    );
    auto tproduction = scalar(0.1355) * ofsTilda * nuTilda + scalar(0.622/0.66666) * ofmagSqrGradNutilda; 
    Foam::volScalarField ofProduction
    (
        Foam::IOobject("ofProduction", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tproduction()
    );
    auto tspCoeff = ofCw1 * ofFw * nuTilda / Foam::sqr(ofdTilda);
    Foam::volScalarField ofspCoeff
    (
        Foam::IOobject("ofspCoeff", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
        tspCoeff()
    );
    // === Mirror state into NeoN ===
    auto& nfU = fieldCollection.registerVector<VolVector>(
        nf::CreateFromFoamField<Foam::volVectorField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = U, .name = "nfU"
        }
    );

    auto& nfp = fieldCollection.registerVector<VolScalar>(
        nf::CreateFromFoamField<Foam::volScalarField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = p, .name = "nfp"
        }
    );

    auto& nfPhi = fieldCollection.registerVector<SurfScalar>(
        nf::CreateFromFoamField<Foam::surfaceScalarField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = phi, .name = "nfPhi"
        }
    );

    auto& nfWallDist = fieldCollection.registerVector<VolScalar>(
        nf::CreateFromFoamField<Foam::volScalarField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = wallDist, .name = "nfWallDist"
        }
    );

    auto& nfNuTilda = fieldCollection.registerVector<VolScalar>(
        nf::CreateFromFoamField<Foam::volScalarField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = nuTilda, .name = "nfNuTilda"
        }
    );

    auto& nfNut = fieldCollection.registerVector<VolScalar>(
        nf::CreateFromFoamField<Foam::volScalarField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = nut, .name = "nfNut"
        }
    );

    auto& nfDelta = fieldCollection.registerVector<VolScalar>(
        nf::CreateFromFoamField<Foam::volScalarField>{
            .exec = rt.exec, .nfMesh = rt.nfMesh, .foamField = delta, .name = "nfDelta"
        }
    );

    // --- Constant nu field in NeoN
    auto volCalcBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Scalar>>(rt.nfMesh);
    auto volCalcVecBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(rt.nfMesh);

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

    NeoN::turbulenceModels::DES::SpalartAllmarasBase saBase(rt.exec, rt.nfMesh);

    auto gradOp = nnfvcc::GaussGreenGrad(rt.exec, rt.nfMesh);

    VolVector gradUx = gradOp.grad(nfU.x());
    VolVector gradUy = gradOp.grad(nfU.y());
    VolVector gradUz = gradOp.grad(nfU.z());
    NeoN::fill(gradUx.boundaryData().value(), Vec3 {0.0, 0.0, 0.0}); 
    NeoN::fill(gradUy.boundaryData().value(), Vec3 {0.0, 0.0, 0.0}); 
    NeoN::fill(gradUz.boundaryData().value(), Vec3 {0.0, 0.0, 0.0});
    
    VolScalar chi(exec, "chi", rt.nfMesh, volCalcBCs);
    VolScalar fv1(exec, "fv1", rt.nfMesh, volCalcBCs);
    VolScalar fv2(exec, "fv2", rt.nfMesh, volCalcBCs);
    VolScalar ft2(exec, "ft2", rt.nfMesh, volCalcBCs);
    VolScalar stilda(exec, "stilda", rt.nfMesh, volCalcBCs);
    VolScalar fw(exec, "fw", rt.nfMesh, volCalcBCs);
    VolScalar dTilda(exec, "dTilda", rt.nfMesh, volCalcBCs);
    VolScalar invSqrdTilda(exec, "invSqrdTilda", rt.nfMesh, volCalcBCs);
    VolScalar omega(exec, "omega", rt.nfMesh, volCalcBCs);
    VolScalar magGradU(exec, "magGradU", rt.nfMesh, volCalcBCs);
    VolVector gradNuTilda(exec, "gradNuTilda", rt.nfMesh, volCalcVecBCs);
    fill(gradNuTilda.internalVector(), Vec3 {0.0, 0.0, 0.0});

    VolScalar nfFusedProduction(exec, "nfFusedProduction", rt.nfMesh, volCalcBCs);
    VolScalar nfFusedSpCoeff(exec, "nfFusedSpCoeff", rt.nfMesh, volCalcBCs);

    NeoN::turbulenceModels::DES::SpalartAllmarasDDES ddes(rt.exec);

    // --- Component chain evaluation
    saBase.chi(chi, nfNuTilda, nfNu);
    nf::compare(chi, ofChi, ApproxScalar(1e-12),false);
    saBase.fv1(fv1, chi);
    nf::compare(fv1, ofFv1, ApproxScalar(1e-12),false);
    saBase.fv2(fv2, chi, fv1);
    nf::compare(fv2, ofFv2, ApproxScalar(1e-12),false);
    const auto& coeffs = saBase.coeffs();
    const auto Cb1 = coeffs.Cb1;
    const auto Cb2 = coeffs.Cb2;
    const auto kappa = coeffs.kappa;
    const auto sigmaNut = coeffs.sigmaNut;
    const auto Cw1 = saBase.cw1();
    const auto Cw2 = coeffs.Cw2;
    const auto Cw3 = coeffs.Cw3;

    REQUIRE(Cw1 == ofCw1);
    saBase.omega(omega, gradUx, gradUy, gradUz);
    nf::compare(omega, ofOmega, ApproxScalar(1e-12),false);
    
    saBase.magGradU(magGradU, gradUx, gradUy, gradUz);
    nf::compare(magGradU, ofMagGradU, ApproxScalar(1e-12),false);  // OpenFOAM BC values are not zero

    ddes.dTilde(dTilda,invSqrdTilda, nfWallDist, nfNuTilda, nfNu, magGradU, nfDelta, chi, fv1);
    nf::compare(dTilda, ofdTilda, ApproxScalar(1e-12),false);
    
    saBase.stilda(stilda, omega, nfNuTilda, dTilda, fv2);
    nf::compare(stilda, ofsTilda, ApproxScalar(1e-12),false);

    saBase.fw(fw, stilda, dTilda, nfNuTilda);
    nf::compare(fw, ofFw, ApproxScalar(1e-12),false);
    saBase.nut(nfNut, nfNuTilda, fv1);
    nf::compare(nfNut, nut, ApproxScalar(1e-12),false);
    
    
    // --- nuEff on faces
    auto surfInterpol = fvcc::SurfaceInterpolation<Scalar>(
        rt.exec, rt.nfMesh, NeoN::TokenList({std::string("linear")})
    );
    auto nuTildaEff = surfInterpol.interpolate(NeoN::scalar(1/sigmaNut)*(nfNuTilda+nfNu));
    nuTildaEff.name = "nuTildaEff";

    fvcc::rotateOldTimes(nfNuTilda);
    gradNuTilda = gradOp.grad(nfNuTilda);
    nf::compare(gradNuTilda, ofgradNutilda, ApproxVector(1e-12),false);
    auto magSqrGradNuTilda = fvcc::magSqr(gradNuTilda);
    nf::compare(magSqrGradNuTilda, ofmagSqrGradNutilda, ApproxScalar(1e-12),false);

    saBase.computeProdSpDDES(nfFusedProduction, nfFusedSpCoeff, nfNuTilda, nfNu,
        omega, nfWallDist, magGradU, nfDelta, magSqrGradNuTilda);
    nf::compare(nfFusedProduction, ofProduction, ApproxScalar(1e-12),false);
    nf::compare(nfFusedSpCoeff, ofspCoeff, ApproxScalar(1e-12),false);

//    auto production = Cb1 * stilda * nfNuTilda + NeoN::scalar(Cb2/sigmaNut) * magSqrGradNuTilda; //ft2 is zero by default
//    nf::compare(production, ofProduction, ApproxScalar(1e-12),false);
//    auto spCoeff = Cw1 * fw * nfNuTilda * invSqrdTilda; // (dTilda * dTilda); // ft2 is zero by default
//    nf::compare(spCoeff, ofspCoeff, ApproxScalar(1e-12),false);
    nf::PDESolver<NeoN::scalar> nuTildaEqn(
        dsl::imp::ddt(nfNuTilda) + dsl::imp::div(nfPhi, nfNuTilda) - NeoN::dsl::imp::laplacian(nuTildaEff, nfNuTilda)
        + dsl::imp::source(nfFusedSpCoeff, nfNuTilda)
        - dsl::exp::source(nfFusedProduction, nfNuTilda),
        nfNuTilda,
        rt
    );
    nuTildaEqn.solve();
    nfNuTilda.correctBoundaryConditions();
    saBase.chi(chi, nfNuTilda, nfNu);
    saBase.fv1(fv1, chi);
    saBase.nut(nfNut, nfNuTilda, fv1);
    nfNut.correctBoundaryConditions();  // nut currently doesn't have the correct BCs
    
    // -----------------------------
    // OpenFOAM nut cross-check
    // -----------------------------
    SECTION("OpenFOAM incompressible::turbulenceModel nut matches (" + execName + ")")
    {
        // Build OpenFOAM turbulence model from dictionaries.
        foamTurb->correct();

	Foam::tmp<Foam::volScalarField> tNutFoam = foamTurb->nut();
	Foam::volScalarField nutFoam(
            Foam::IOobject("nutFoam", runTime.timeName(), mesh, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
            tNutFoam()
        );

        nf::compare(nfNut, nutFoam, ApproxScalar(1e-7));
    }  
} 

