// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// NeoFOAM against OpenFOAM on a 387-cell region cut out of the TUD campus case (subsetMesh around a
// Buildings wall cell, fields from a converged neoSimpleFoam state, cut faces fixed). In that wall
// cell k is ~1e-3 of its neighbours' and the pinned epsilon tiny. In the full case one SIMPLE
// iteration took k there to ~0.1 in NeoFOAM but ~0.002 in OpenFOAM, and nut = Cmu k^2/epsilon to
// ~3500x OpenFOAM's, which seeded the k-epsilon blow-ups.
//
//   - The turbulence step on frozen U/phi: given the same U and phi both codes take k to ~0.1, so
//     the difference comes from the momentum/pressure step that produces U and phi.
//   - The momentum terms on the same frozen state, one at a time, as the per-volume cell residual
//     (A U - b)/V of each assembled term against OpenFOAM's (M & U).

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

namespace
{
// The Buildings wall cell the blow-ups started from (cell 9946048 of the full mesh).
constexpr Foam::label hotCell = 218;

template<typename MatrixValue>
NeoN::Vec3 times(const MatrixValue& a, const NeoN::Vec3& x)
{
    if constexpr (std::is_same_v<MatrixValue, NeoN::scalar>)
    {
        return a * x;
    }
    else
    {
        return NeoN::Vec3(a[0] * x[0], a[1] * x[1], a[2] * x[2]);
    }
}

// Per-volume cell residual (A x - b)/V of an assembled single-rank system: the NeoFOAM counterpart
// of OpenFOAM's (M & psi).
template<typename LinearSystem>
std::vector<NeoN::Vec3> cellResidual(
    const LinearSystem& ls, const NeoN::Vector<NeoN::Vec3>& x, const NeoN::Vector<NeoN::scalar>& vol
)
{
    const auto valsH = ls.matrix().values().copyToHost();
    const auto rowsH = ls.matrix().sparsity()->rowOffs().copyToHost();
    const auto colsH = ls.matrix().sparsity()->colIdxs().copyToHost();
    const auto rhsH = ls.rhs().copyToHost();
    const auto xH = x.copyToHost();
    const auto volH = vol.copyToHost();
    const auto [vals, rows, cols, rhs, xv, v] =
        NeoN::views(valsH, rowsH, colsH, rhsH, xH, volH);
    std::vector<NeoN::Vec3> r(static_cast<std::size_t>(xv.size()));
    for (NeoN::localIdx i = 0; i < xv.size(); ++i)
    {
        NeoN::Vec3 ax(0, 0, 0);
        for (auto j = rows[i]; j < rows[i + 1]; ++j)
        {
            ax += times(vals[j], xv[cols[j]]);
        }
        r[static_cast<std::size_t>(i)] = (ax - rhs[i]) * (1.0 / v[i]);
    }
    return r;
}

// Compare a NeoFOAM per-cell vector field with an OpenFOAM one up to a global sign (the two
// codes' assembled systems need not share one). Prints the worst cell and the listed cells;
// returns max|nf - s of| / max|of| for the better-fitting sign s.
NeoN::scalar compareTerm(
    const std::string& name,
    const std::vector<NeoN::Vec3>& nf,
    const Foam::vectorField& of,
    const std::vector<Foam::label>& cells
)
{
    NeoN::scalar ofMax = 0;
    for (const auto& o : of) ofMax = Foam::max(ofMax, Foam::mag(o));
    auto maxDiff = [&](NeoN::scalar s, Foam::label& at)
    {
        NeoN::scalar m = 0;
        forAll(of, c)
        {
            const auto& n = nf[static_cast<std::size_t>(c)];
            const NeoN::scalar d = Foam::mag(Foam::vector(n[0], n[1], n[2]) - s * of[c]);
            if (d > m)
            {
                m = d;
                at = c;
            }
        }
        return m;
    };
    Foam::label atPlus = -1, atMinus = -1;
    const NeoN::scalar dPlus = maxDiff(1, atPlus), dMinus = maxDiff(-1, atMinus);
    const NeoN::scalar s = dPlus <= dMinus ? 1 : -1;
    const Foam::label at = dPlus <= dMinus ? atPlus : atMinus;
    const NeoN::scalar rel = Foam::min(dPlus, dMinus) / Foam::max(ofMax, NeoN::scalar(1e-300));
    std::cout << "TERM " << name << ": sign " << s << ", max|d|/max|of| " << rel << " at cell "
              << at << " (max|of| " << ofMax << ")\n";
    std::vector<Foam::label> show(cells);
    show.push_back(at);
    for (const auto c : show)
    {
        const auto& n = nf[static_cast<std::size_t>(c)];
        const Foam::vector o = s * of[c];
        std::cout << "  cell " << c << " neo (" << n[0] << " " << n[1] << " " << n[2] << ") of ("
                  << o.x() << " " << o.y() << " " << o.z() << ")\n";
    }
    return rel;
}
} // namespace

TEST_CASE("kEpsilon: low-k wall cell next to high-k cells matches OpenFOAM")
{
    Foam::Time& runTime = *timePtr;

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("k") = nf::mapFvSolution(solverDict.subDict("k"));
    solverDict.subDict("epsilon") = nf::mapFvSolution(solverDict.subDict("epsilon"));
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(U)
    );

    Foam::singlePhaseTransportModel transport(U, phi);
    auto tnu = transport.nu();

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();

    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    REQUIRE(mesh.nCells() == 387);
    const Foam::scalar kStart = ofK[hotCell];

    const Foam::volScalarField wallDist = nf::makeNearWallDistField(mesh);

    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfEps = NeoFOAM::constructAndRegister(fieldCollection, rt, ofEps, false);

    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, tnu(), ofNut);

    nf::KEpsilon turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);
    turbNF.setWallRoughness(
        NeoN::Vector<NeoN::scalar>(rt.exec, nf::readEpsilonWallRoughness(ofEps, rt.nfMesh))
    );
    turbNF.validate(nfU, nfK, nfEps, nfNut);

    turbNF.correct(nfU, nfPhi, nfK, nfEps, nfNut, rt);
    foamTurb->correct();

    const Foam::volScalarField& kFoam = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& epsFoam = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& nutFoam = mesh.lookupObject<Foam::volScalarField>("nut");

    const auto kHost = nfK.internalVector().copyToHost();
    const auto nutHost = nfNut.internalVector().copyToHost();
    INFO(
        "hot cell " << hotCell << ": k start " << kStart << ", NeoFOAM " << kHost.view()[hotCell]
                    << ", OpenFOAM " << kFoam[hotCell] << "; nut NeoFOAM "
                    << nutHost.view()[hotCell] << ", OpenFOAM " << nutFoam[hotCell]
    );

    // The cell the blow-ups started from, on its own so a failure names it.
    REQUIRE(kHost.view()[hotCell] == Catch::Approx(kFoam[hotCell]).epsilon(1e-3));

    REQUIRE_THAT(nfEps, EqualsInternal(epsFoam, ApproxScalar(1e-6)));
    REQUIRE_THAT(nfK, EqualsInternal(kFoam, ApproxScalar(1e-5)));
    REQUIRE_THAT(nfNut, EqualsInternal(nutFoam, ApproxScalar(1e-5)));
}

TEST_CASE("momentum terms at a Buildings wall region match OpenFOAM")
{
    Foam::Time& runTime = *timePtr;

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(U)
    );

    Foam::singlePhaseTransportModel transport(U, phi);
    auto tnu = transport.nu();
    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();

    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");
    const Foam::volScalarField wallDist = nf::makeNearWallDistField(mesh);

    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfEps = NeoFOAM::constructAndRegister(fieldCollection, rt, ofEps, false);
    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, tnu(), ofNut);

    nf::KEpsilon turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);
    turbNF.setWallRoughness(
        NeoN::Vector<NeoN::scalar>(rt.exec, nf::readEpsilonWallRoughness(ofEps, rt.nfMesh))
    );
    turbNF.validate(nfU, nfK, nfEps, nfNut);
    turbNF.updateGradU(nfU);

    // The cells to report: the hot cell and the Buildings wall cells.
    std::vector<Foam::label> cells {hotCell};
    {
        const auto& wallCells = mesh.boundary()[mesh.boundaryMesh().findPatchID("Buildings")].faceCells();
        for (const auto c : wallCells)
            if (std::find(cells.begin(), cells.end(), c) == cells.end() && cells.size() < 6)
                cells.push_back(c);
    }

    const auto& vol = rt.nfMesh.cellVolumes();
    auto residualOf = [&](auto&& expr)
    {
        nf::PDE<NeoN::Vec3> pde(dsl::Expression<NeoN::Vec3>(expr), nfU, rt);
        pde.assemble();
        return cellResidual(pde.linearSystem(), nfU.internalVector(), vol);
    };

    // nuEff on the faces, as both momentum equations see it.
    const Foam::volScalarField ofNuEff(foamTurb->nuEff());

    std::cout << "=== momentum terms (" << execName << ")\n";

    const auto relDiv = compareTerm(
        "div(phi,U)",
        residualOf(dsl::imp::div(nfPhi, nfU)),
        (Foam::fvVectorMatrix(fvm::div(phi, U)) & U)().primitiveField(),
        cells
    );
    const auto relLap = compareTerm(
        "laplacian(nuEff,U)",
        residualOf(dsl::imp::laplacian(turbNF.nuEff(), nfU)),
        (Foam::fvVectorMatrix(fvm::laplacian(ofNuEff, U)) & U)().primitiveField(),
        cells
    );
    const auto relStress = compareTerm(
        "viscousStress",
        residualOf(dsl::exp::viscousStress(nfNu, nfNut, turbNF.gradU())),
        fvc::div(ofNuEff * Foam::dev2(Foam::T(fvc::grad(U))))().primitiveField(),
        cells
    );
    const auto relGradP = compareTerm(
        "grad(p)",
        residualOf(dsl::exp::grad(nfP)),
        fvc::grad(p)().primitiveField(),
        cells
    );

    // grad(U) itself, which feeds the explicit stress (row-wise, as three vectors). NeoN stores
    // dU_i/dx_j at (i, j), OpenFOAM at (j, i): NeoN's row i is OpenFOAM's column i.
    NeoN::scalar relGradU = 0;
    {
        const Foam::volTensorField ofGradU(fvc::grad(U));
        const auto gH = turbNF.gradU().internalVector().copyToHost();
        const auto g = gH.view();
        for (int row = 0; row < 3; ++row)
        {
            std::vector<NeoN::Vec3> nfRow(static_cast<std::size_t>(g.size()));
            Foam::vectorField ofRow(ofGradU.size());
            forAll(ofGradU, c)
            {
                nfRow[static_cast<std::size_t>(c)] =
                    NeoN::Vec3(g[c](row, 0), g[c](row, 1), g[c](row, 2));
                ofRow[c] = Foam::vector(
                    ofGradU[c].component(row),
                    ofGradU[c].component(3 + row),
                    ofGradU[c].component(6 + row)
                );
            }
            relGradU = Foam::max(
                relGradU, compareTerm("grad(U) row " + std::to_string(row), nfRow, ofRow, cells)
            );
        }
    }

    CHECK(relDiv < 1e-8);
    CHECK(relLap < 1e-8);
    CHECK(relStress < 1e-8);
    CHECK(relGradP < 1e-8);
    CHECK(relGradU < 1e-8);
}
