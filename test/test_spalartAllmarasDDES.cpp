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

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
    solverDict.subDict("nuTilda") = nf::mapFvSolution(solverDict.subDict("nuTilda"));
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);
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

    Foam::volScalarField nearWallDist(
        Foam::IOobject(
            "nearWallDist",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("zero", Foam::dimLength, scalar(0.0))
    );
    for (Foam::label patchi = 0; patchi < nPatches; ++patchi)
    {
        const Foam::scalarField& ofY = turb.y()[patchi];
        Foam::fvPatchScalarField& yPatch = nearWallDist.boundaryFieldRef()[patchi];

        const Foam::label nFaces = yPatch.size();

        for (Foam::label facei = 0; facei < nFaces; ++facei)
        {
            yPatch[facei] = ofY[facei];
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

    auto tviscousStress = -fvc::div(foamTurb->nuEff() * Foam::dev2(Foam::T(fvc::grad(U))));
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

    // === Mirror state into NeoN ===
    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfNuTilda = NeoFOAM::constructAndRegister(fieldCollection, rt, nuTilda, false);

     auto [nfWallDist, nfDelta, nut, nfNearWallDist] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, delta, ofNut, nearWallDist);

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
    fvcc::TensorVecField G {
        VolVector(exec, "gradUx", rt.nfMesh, volCalcVecBCs),
        VolVector(exec, "gradUy", rt.nfMesh, volCalcVecBCs),
        VolVector(exec, "gradUz", rt.nfMesh, volCalcVecBCs)
    };
    gradOp.grad(nfU, G);
    nf::compare(G.Tx, ofGradUx, ApproxVector(1e-10), true);
    nf::compare(G.Ty, ofGradUy, ApproxVector(1e-10), true);
    nf::compare(G.Tz, ofGradUz, ApproxVector(1e-10), true);

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

    REQUIRE(Cw1 == ofCw1);

    auto surfInterpol = fvcc::SurfaceInterpolation<Scalar>(
        rt.exec,
        rt.nfMesh,
        NeoN::TokenList({std::string("linear")})
    );
    surfInterpol.interpolate(nfNu, nfSurfNu);

    saBase.correctNut(nut, nfSurfNut, nfSurfNuEff, nfNuTilda, nfNu, nfSurfNu, nfU, nfNearWallDist);
    nf::compare(nut, ofNut, ApproxScalar(1e-12));
    nf::compare(nfNuTilda, nuTilda, ApproxScalar(1e-12));
    saBase.calcNuTildaDiffusionCoeff(nfNuTilda, nfSurfNu, nfSurfNuTilda, nuTildaEff);

    fvcc::rotateOldTimes(nfU);
    fvcc::rotateOldTimes(nfPhi);
    fvcc::rotateOldTimes(nfNuTilda);
    gradOp.grad(nfNuTilda, gradNuTilda);
    nf::compare(gradNuTilda, ofgradNutilda, ApproxVector(1e-12));
    saBase.calcMagSqrVec(magSqrGradNuTilda, gradNuTilda);
    nf::compare(magSqrGradNuTilda, ofmagSqrGradNutilda, ApproxScalar(1e-12), false);

    saBase.computeProdSpDDES(
        nfFusedProduction,
        nfFusedSpCoeff,
        nfNuTilda,
        nfNu,
        G.Tx,
        G.Ty,
        G.Tz,
        nfWallDist,
        nfDelta,
        magSqrGradNuTilda
    );
    nf::compare(nfFusedProduction, ofProduction, ApproxScalar(1e-12), false);
    nf::compare(nfFusedSpCoeff, ofspCoeff, ApproxScalar(1e-12), false);

    NeoN::TokenList interpolationScheme;
    interpolationScheme.insert(std::string("linear"));
    interpolationScheme.insert(std::string("uncorrected"));
    fvcc::GaussViscousStress opVisc(exec, rt.nfMesh, interpolationScheme);
    auto nfViscousStress = opVisc.viscousStress(nfNu, nut, G, dsl::Coeff(1.0));
    nf::compare(nfViscousStress, ofViscousStress, ApproxVector(1e-12), false);
    // nf::compare(nfU, U, ApproxVector(1e-12));
    nf::PDESolver<NeoN::Vec3> UEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfSurfNuEff, nfU)
            + dsl::exp::viscousStress(nfNu, nut, G),
        nfU,
        rt
    );
    UEqn.solve(-1.0 * dsl::exp::grad(nfP));
    nfU.correctBoundaryConditions();

    gradOp.grad(nfU, G);
    gradOp.grad(nfNuTilda, gradNuTilda);
    saBase.calcMagSqrVec(magSqrGradNuTilda, gradNuTilda);
    saBase.computeProdSpDDES(
        nfFusedProduction,
        nfFusedSpCoeff,
        nfNuTilda,
        nfNu,
        G.Tx,
        G.Ty,
        G.Tz,
        nfWallDist,
        nfDelta,
        magSqrGradNuTilda
    );

    nf::PDESolver<NeoN::scalar> nuTildaEqn(
        dsl::imp::ddt(nfNuTilda) + dsl::imp::div(nfPhi, nfNuTilda)
            - dsl::imp::laplacian(nuTildaEff, nfNuTilda)
            + dsl::imp::source(nfFusedSpCoeff, nfNuTilda) - dsl::exp::sourceU(nfFusedProduction),
        nfNuTilda,
        rt
    );
    nuTildaEqn.solve();
    saBase.calcNuTildaDiffusionCoeff(nfNuTilda, nfSurfNu, nfSurfNuTilda, nuTildaEff);
    saBase.correctNut(nut, nfSurfNut, nfSurfNuEff, nfNuTilda, nfNu, nfSurfNu, nfU, nfNearWallDist);

    Foam::fvVectorMatrix ofUEqn(
        Foam::fvm::ddt(U) + Foam::fvm::div(phi, U) + foamTurb->divDevReff(U)
    );
    Foam::solve(ofUEqn == -fvc::grad(p));
    U.correctBoundaryConditions();
    nf::compare(nfU, U, ApproxVector(1e-10));

    foamTurb->correct();

    const Foam::volScalarField& nutFoam = mesh.lookupObject<Foam::volScalarField>("nut");
    const Foam::volScalarField& nuTildaFoam = mesh.lookupObject<Foam::volScalarField>("nuTilda");
    nf::compare(nfNuTilda, nuTildaFoam, ApproxScalar(1e-12));
    nf::compare(nut, nutFoam, ApproxScalar(1e-12));
}

TEST_CASE("SA-DDES: NeoFOAM wrapper validate() + correct() matches OpenFOAM")
{
    Foam::Time& runTime = *timePtr;

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
    solverDict.subDict("nuTilda") = nf::mapFvSolution(solverDict.subDict("nuTilda"));
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);
    const NeoN::Dictionary controlDict = nf::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");
    runTime.setDeltaT(dt);

    // --- OpenFOAM fields ---
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
        fvc::flux(U)
    );

    Foam::singlePhaseTransportModel transport(U, phi);
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

    const Foam::label nPatches = mesh.boundary().size();
    Foam::volScalarField nearWallDist(
        Foam::IOobject(
            "nearWallDist",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("zero", Foam::dimLength, Foam::scalar(0.0))
    );
    for (Foam::label patchi = 0; patchi < nPatches; ++patchi)
    {
        const Foam::scalarField& ofY = turb.y()[patchi];
        Foam::fvPatchScalarField& yPatch = nearWallDist.boundaryFieldRef()[patchi];
        for (Foam::label facei = 0; facei < yPatch.size(); ++facei)
            yPatch[facei] = ofY[facei];
    }

    // --- NeoFOAM fields ---
    auto& nfU       = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP       = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi     = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfNuTilda = NeoFOAM::constructAndRegister(fieldCollection, rt, nuTilda, false);

    auto [nfWallDist, nfDelta, nut, nfNearWallDist] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, delta, ofNut, nearWallDist);

    auto volCalcBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Scalar>>(rt.nfMesh);
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

    // --- Wrapper: validate then one time step ---
    nf::SpalartAllmarasDDES turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist, nfNearWallDist, nfDelta);
    turbNF.validate(nfU, nfNuTilda, nut);

    fvcc::rotateOldTimes(nfU);
    fvcc::rotateOldTimes(nfPhi);
    fvcc::rotateOldTimes(nfNuTilda);

    nf::PDESolver<NeoN::Vec3> UEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
            - dsl::imp::laplacian(turbNF.nuEff(), nfU)
            + dsl::exp::viscousStress(nfNu, nut, turbNF.gradU()),
        nfU,
        rt
    );
    UEqn.solve(-1.0 * dsl::exp::grad(nfP));
    nfU.correctBoundaryConditions();

    turbNF.correct(nfU, nfPhi, nfNuTilda, nut, rt);

    // --- OpenFOAM reference ---
    Foam::fvVectorMatrix ofUEqn(
        Foam::fvm::ddt(U) + Foam::fvm::div(phi, U) + foamTurb->divDevReff(U)
    );
    Foam::solve(ofUEqn == -fvc::grad(p));
    U.correctBoundaryConditions();
    foamTurb->correct();

    // --- Comparisons ---
    nf::compare(nfU, U, ApproxVector(1e-10));
    const Foam::volScalarField& nuTildaFoam2 = mesh.lookupObject<Foam::volScalarField>("nuTilda");
    const Foam::volScalarField& nutFoam2 = mesh.lookupObject<Foam::volScalarField>("nut");
    nf::compare(nfNuTilda, nuTildaFoam2, ApproxScalar(1e-12));
    nf::compare(nut, nutFoam2, ApproxScalar(1e-12));
}
