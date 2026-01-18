// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;
namespace dsl = NeoN::dsl;
namespace la = NeoN::la;

#include "fvc.H"
#include "fvm.H"
#include "fvMatrices.H"

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

// Fused implicit kernel: ddt + div + laplacian in single assembly pass
template<typename ValueType>
void fusedImplicitAssembly(
    la::LinearSystem<ValueType, NeoN::localIdx>& ls,
    const fvcc::VolumeField<ValueType>& field,
    const NeoN::Vector<ValueType>& oldTimeField,
    const fvcc::SurfaceField<NeoN::scalar>& faceFlux,
    const NeoN::Vector<NeoN::scalar>& weights,
    const fvcc::SurfaceField<NeoN::scalar>& gamma,
    const NeoN::Vector<NeoN::scalar>& deltaCoeffs,
    const la::SparsityPattern& sparsityPattern,
    NeoN::scalar dt
)
{
    using namespace NeoN;
    const auto& mesh = field.mesh();
    const auto exec = field.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();

    // Get views for all data
    const auto [vol, magFaceArea, owner, neighbour, surfFaceCells] = views(
        mesh.cellVolumes(),
        mesh.magFaceAreas(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells()
    );

    const auto [diagOffs, ownOffs, neiOffs] = views(
        sparsityPattern.diagOffset(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset()
    );

    const auto [faceFluxV, weightsV, gammaV, deltaCoeffsV, oldVector] = views(
        faceFlux.internalVector(),
        weights,
        gamma.internalVector(),
        deltaCoeffs,
        oldTimeField
    );

    auto [matrix, rhs] = ls.view();

    // Pass 1: Cell-based ddt assembly (BDF1)
    const scalar a0 = 1.0 / dt;
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto coeff = a0 * vol[celli];
            matrix.values[idx] += coeff * one<ValueType>();
            rhs[celli] += coeff * oldVector[celli];
        },
        "fusedKernel::ddtPass"
    );

    // Pass 2: Face-based div + laplacian assembly
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto flux = faceFluxV[facei];
            const auto weight = weightsV[facei];
            const auto own = owner[facei];
            const auto nei = neighbour[facei];

            const auto rowOwnStart = matrix.rowOffs[own];
            const auto rowNeiStart = matrix.rowOffs[nei];

            // DIV + LAPLACIAN contributions (combined)
            const auto lapFlux = deltaCoeffsV[facei] * gammaV[facei] * magFaceArea[facei];

            // Combined contributions for neighbour column in owner row
            const auto combinedValueNei = (-weight * flux + lapFlux) * one<ValueType>();
            matrix.values[rowNeiStart + neiOffs[facei]] += combinedValueNei;
            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diagOffs[own]], combinedValueNei);
            // matrix.values[rowOwnStart + diagOffs[own]] -= combinedValueNei;


            // Combined contributions for owner column in neighbour row
            const auto combinedValueOwn = (flux * (1.0 - weight) + lapFlux) * one<ValueType>();
            matrix.values[rowOwnStart + ownOffs[facei]] += combinedValueOwn;
            Kokkos::atomic_sub(&matrix.values[rowNeiStart + diagOffs[nei]], combinedValueOwn);
            // matrix.values[rowNeiStart + diagOffs[nei]] -= combinedValueOwn;
        },
        "fusedKernel::divLaplacianPass"
    );

    // Pass 3: Boundary contributions
    const auto [refGradient, value, valueFraction, refValue, bDeltaCoeffs] = views(
        field.boundaryData().refGrad(),
        field.boundaryData().value(),
        field.boundaryData().valueFraction(),
        field.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto& bcCoeffs =
        ls.auxiliaryCoefficients().template get<la::BoundaryCoefficients<ValueType, localIdx>>(
            "boundaryCoefficients"
        );
    auto [boundValues, rhsBoundValues] = views(bcCoeffs.matrixValues, bcCoeffs.rhsValues);

    parallelFor(
        exec,
        {nInternalFaces, faceFluxV.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto bcfacei = facei - nInternalFaces;
            const auto own = surfFaceCells[bcfacei];
            const auto rowOwnStart = matrix.rowOffs[own];

            const auto valFrac1 = valueFraction[bcfacei];
            const auto valFrac2 = 1.0 - valFrac1;

            // Combined DIV + LAPLACIAN boundary contribution
            const auto divFlux = weightsV[facei] * faceFluxV[facei];
            const auto lapFlux = gammaV[facei] * magFaceArea[facei];

            const auto combinedValueMat =
                (divFlux * valFrac2 - lapFlux * valFrac1 * deltaCoeffsV[facei]) * one<ValueType>();
            const auto combinedValueRhs =
                (divFlux * valFrac1 * refValue[bcfacei]
                 + valFrac2 * refGradient[bcfacei] * (1.0 / bDeltaCoeffs[bcfacei]))
                + lapFlux
                      * (valFrac1 * deltaCoeffsV[facei] * refValue[bcfacei]
                         + valFrac2 * refGradient[bcfacei]);

            // Single atomic operations
            Kokkos::atomic_add(&matrix.values[rowOwnStart + diagOffs[own]], combinedValueMat);
            Kokkos::atomic_sub(&rhs[own], combinedValueRhs);

            boundValues[bcfacei] = combinedValueMat;
            rhsBoundValues[bcfacei] = combinedValueRhs;
        },
        "fusedKernel::boundaryPass"
    );
}

TEST_CASE("advection-diffusion-equation_scalar")
{
    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofT = randomScalarField(runTime, mesh, "T");
    Foam::surfaceScalarField ofPhi(
        Foam::IOobject("phi", "0", mesh, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
        mesh,
        Foam::dimensionedScalar("phi", Foam::dimensionSet(0, 3, -1, 0, 0), 0.0)
    );
    forAll(ofPhi, facei)
    {
        ofPhi[facei] = facei;
    }

    Foam::surfaceScalarField ofGamma(
        Foam::IOobject("Gamma", "0", mesh, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
        mesh,
        Foam::dimensionedScalar("phi", Foam::dimensionSet(0, 2, -1, 0, 0), 0.0)
    );

    SECTION("OpenFOAM")
    {
        SECTION("explicit-time-integration")
        {

            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvScalarMatrix advectDiffEqn(
                    Foam::fvm::ddt(ofT) + Foam::fvc::div(ofPhi, ofT)
                    - Foam::fvc::laplacian(ofGamma, ofT)
                );
                return advectDiffEqn;
            };
        }

        SECTION("implicit-time-integration")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvScalarMatrix advectDiffEqn(
                    Foam::fvm::ddt(ofT) + Foam::fvm::div(ofPhi, ofT)
                    - Foam::fvm::laplacian(ofGamma, ofT)
                );
                return advectDiffEqn;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        auto rt = nf::createAdapterRunTime(runTime, exec);

        auto& vectorCollection = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        fvcc::VolumeField<NeoN::scalar>& nfT =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofT,
                    .name = "nfT"
                }
            );
        nfT.correctBoundaryConditions();

        auto& nfOldT = fvcc::oldTime(nfT);
        nfOldT.internalVector() = nfT.internalVector();

        auto nfPhi = nf::constructFrom(rt.exec, rt.nfMesh, ofPhi);
        auto nfGamma = nf::constructFrom(rt.exec, rt.nfMesh, ofGamma);

        SECTION(std::string("explicit-time-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary({{std::string("ddt(nfT)"), std::string("BDF1")}})
            );
            rt.fvSchemesDict.insert(
                std::string("divSchemes"),
                NeoN::Dictionary(
                    {{std::string("div(phi,nfT)"),
                      NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("laplacianSchemes"),
                NeoN::Dictionary(
                    {{std::string("laplacian(Gamma,nfT)"),
                      NeoN::TokenList(
                          {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                      )}}
                )
            );

            BENCHMARK(std::string(execName))
            {
                auto eqn = nf::PDESolver(
                    dsl::imp::ddt(nfT) + dsl::exp::div(nfPhi, nfT)
                        - dsl::exp::laplacian(nfGamma, nfT),
                    nfT,
                    rt
                );
                return eqn.assemble();
            };
        }

        SECTION(std::string("implicit-time-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary({{std::string("ddt(nfT)"), std::string("BDF1")}})
            );
            rt.fvSchemesDict.insert(
                std::string("divSchemes"),
                NeoN::Dictionary(
                    {{std::string("div(phi,nfT)"),
                      NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("laplacianSchemes"),
                NeoN::Dictionary(
                    {{std::string("laplacian(Gamma,nfT)"),
                      NeoN::TokenList(
                          {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                      )}}
                )
            );

            BENCHMARK(std::string(execName))
            {
                auto eqn = nf::PDESolver(
                    dsl::imp::ddt(nfT) + dsl::imp::div(nfPhi, nfT)
                        - dsl::imp::laplacian(nfGamma, nfT),
                    nfT,
                    rt
                );
                return eqn.assemble();
            };
        }

        SECTION(std::string("fused-implicit-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary({{std::string("ddt(nfT)"), std::string("BDF1")}})
            );
            rt.fvSchemesDict.insert(
                std::string("divSchemes"),
                NeoN::Dictionary(
                    {{std::string("div(phi,nfT)"),
                      NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("laplacianSchemes"),
                NeoN::Dictionary(
                    {{std::string("laplacian(Gamma,nfT)"),
                      NeoN::TokenList(
                          {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                      )}}
                )
            );

            // Setup required operators and interpolation
            auto divTokens =
                rt.fvSchemesDict.subDict("divSchemes").get<NeoN::TokenList>("div(phi,nfT)");
            std::string interpScheme = divTokens.get<std::string>(1); // "upwind"
            auto surfInterp = fvcc::SurfaceInterpolationFactory<NeoN::scalar>::create(
                exec,
                rt.nfMesh,
                NeoN::TokenList({interpScheme})
            );
            fvcc::SurfaceField<NeoN::scalar> weights(
                exec,
                "weights",
                rt.nfMesh,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh)
            );
            surfInterp->weight(nfPhi, nfT, weights);

            auto lapTokens = rt.fvSchemesDict.subDict("laplacianSchemes")
                                 .get<NeoN::TokenList>("laplacian(Gamma,nfT)");
            std::string correctionScheme = lapTokens.get<std::string>(2); // "uncorrected"
            auto faceNormalGrad = fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(
                exec,
                rt.nfMesh,
                NeoN::TokenList({correctionScheme})
            );

            const auto& sparsityPattern = la::SparsityPattern::readOrCreate(rt.nfMesh);
            const scalar dt = 1.0; // Fixed timestep for benchmark

            BENCHMARK(std::string(execName) + "-fused")
            {
                // Create empty linear system
                auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
                    rt.nfMesh,
                    sparsityPattern
                );

                // Call fused kernel
                fusedImplicitAssembly<NeoN::scalar>(
                    ls,
                    nfT,
                    fvcc::oldTime(nfT).internalVector(),
                    nfPhi,
                    weights.internalVector(),
                    nfGamma,
                    faceNormalGrad->deltaCoeffs().internalVector(),
                    sparsityPattern,
                    dt
                );

                return ls;
            };
        }
    }
}
