// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// NeoFOAM against OpenFOAM on a 428-cell region cut out of the TUD campus case (subsetMesh, 10 m
// sphere around a Buildings wall-function cell, cell 8483719 of the full mesh), fields from
// OpenFOAM's converged simpleFoam state, cut faces fixed. In the full case NeoFOAM restarted from
// that state drove k in the wall cell from 0.13 to 0.23 within five SIMPLE iterations and then
// into a period-2 flip-flop of k and nut (nut 1e-5 <-> 6), while OpenFOAM keeps k at 0.124. The
// flip-flop held the Uz residual at ~4e-4 and seeded the k-epsilon blow-ups.
//
// With identical wall-function inputs and identical k matrices one turbulence step still took k
// to 0.185 (OpenFOAM 0.129) in the wall cell and to 1.78 (1.27) in a wall cell whose bounded
// convection left a negative diagonal. Two causes:
//   - bound() refilled undershooting cells from a neighbourhood average over stale boundary
//     values; fvMatrix::solve() evaluates the boundary before OpenFOAM's bound() runs;
//   - NeoN's equation relaxation kept a negative diagonal negative, where fvMatrix::relax()
//     makes every relaxed diagonal positive (D = max(|D|, sumOff)).
//
// The test cases compare, on the frozen U/phi:
//   - the wall-function inputs face by face (y, deltaCoeffs, |dU/dn|, nut_w) and the production;
//   - one turbulence step (k, epsilon, nut);
//   - the k and epsilon transport terms, including the diagonal/off-diagonal split that
//     relaxation acts on.
// (Several steps on the frozen flow are no test: with U and the cut faces frozen OpenFOAM itself
// flip-flops after five steps.)

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

#include "NeoFOAM/compatibility/fvSolution.hpp"

namespace fvc = Foam::fvc;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

namespace
{
// The Buildings wall-function cell (cell 8483719 of the full mesh; one Buildings face).
constexpr Foam::label hotCell = 40;

std::vector<NeoN::scalar> toStd(const NeoN::Vector<NeoN::scalar>& v)
{
    const auto h = v.copyToHost();
    const auto view = h.view();
    std::vector<NeoN::scalar> out(static_cast<std::size_t>(view.size()));
    for (NeoN::localIdx i = 0; i < view.size(); ++i) out[static_cast<std::size_t>(i)] = view[i];
    return out;
}

std::vector<NeoN::Vec3> toStd(const NeoN::Vector<NeoN::Vec3>& v)
{
    const auto h = v.copyToHost();
    const auto view = h.view();
    std::vector<NeoN::Vec3> out(static_cast<std::size_t>(view.size()));
    for (NeoN::localIdx i = 0; i < view.size(); ++i) out[static_cast<std::size_t>(i)] = view[i];
    return out;
}

// max|nf - of| / max|of| over the given values; prints it and the listed indices.
NeoN::scalar compare(
    const std::string& name,
    const std::vector<NeoN::scalar>& nfValues,
    const Foam::scalarField& of,
    const std::vector<Foam::label>& show
)
{
    REQUIRE(nfValues.size() == static_cast<std::size_t>(of.size()));
    NeoN::scalar ofMax = 0, dMax = 0;
    Foam::label at = -1;
    forAll(of, i)
    {
        ofMax = Foam::max(ofMax, Foam::mag(of[i]));
        const NeoN::scalar d = Foam::mag(nfValues[static_cast<std::size_t>(i)] - of[i]);
        if (d > dMax)
        {
            dMax = d;
            at = i;
        }
    }
    const NeoN::scalar rel = dMax / Foam::max(ofMax, NeoN::scalar(1e-300));
    std::cout << "TERM " << name << ": max|d|/max|of| " << rel << " at " << at << " (max|of| "
              << ofMax << ")\n";
    std::vector<Foam::label> idx(show);
    if (at >= 0) idx.push_back(at);
    for (const auto i : idx)
    {
        std::cout << "  [" << i << "] neo " << nfValues[static_cast<std::size_t>(i)] << " of "
                  << of[i] << "\n";
    }
    return rel;
}

// compare() for vector fields: max|nf - of| / max|of| with |.| the vector magnitude.
NeoN::scalar compare(
    const std::string& name,
    const std::vector<NeoN::Vec3>& nfValues,
    const Foam::vectorField& of,
    const std::vector<Foam::label>& show
)
{
    REQUIRE(nfValues.size() == static_cast<std::size_t>(of.size()));
    NeoN::scalar ofMax = 0, dMax = 0;
    Foam::label at = -1;
    auto asFoam = [](const NeoN::Vec3& v) { return Foam::vector(v[0], v[1], v[2]); };
    forAll(of, i)
    {
        ofMax = Foam::max(ofMax, Foam::mag(of[i]));
        const NeoN::scalar d = Foam::mag(asFoam(nfValues[static_cast<std::size_t>(i)]) - of[i]);
        if (d > dMax)
        {
            dMax = d;
            at = i;
        }
    }
    const NeoN::scalar rel = dMax / Foam::max(ofMax, NeoN::scalar(1e-300));
    std::cout << "TERM " << name << ": max|d|/max|of| " << rel << " at " << at << " (max|of| "
              << ofMax << ")\n";
    std::vector<Foam::label> idx(show);
    if (at >= 0) idx.push_back(at);
    for (const auto i : idx)
    {
        const auto& n = nfValues[static_cast<std::size_t>(i)];
        std::cout << "  [" << i << "] neo (" << n[0] << " " << n[1] << " " << n[2] << ") of ("
                  << of[i].x() << " " << of[i].y() << " " << of[i].z() << ")\n";
    }
    return rel;
}

// The first n entries (the internal faces of a NeoN surface field).
template<typename T>
std::vector<T> head(std::vector<T> v, Foam::label n)
{
    v.resize(static_cast<std::size_t>(n));
    return v;
}

// Per-volume cell residual (A x - b)/V of an assembled vector system (scalar matrix, Vec3 rhs).
template<typename LinearSystem>
std::vector<NeoN::Vec3> cellResidualVec(
    const LinearSystem& ls,
    const NeoN::Vector<NeoN::Vec3>& x,
    const NeoN::Vector<NeoN::scalar>& vol
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
        for (auto j = rows[i]; j < rows[i + 1]; ++j) ax += vals[j] * xv[cols[j]];
        r[static_cast<std::size_t>(i)] = (ax - rhs[i]) * (1.0 / v[i]);
    }
    return r;
}

// Per-volume cell residual (A x - b)/V of an assembled scalar system: the NeoFOAM counterpart of
// OpenFOAM's (M & psi).
template<typename LinearSystem>
std::vector<NeoN::scalar> cellResidual(
    const LinearSystem& ls,
    const NeoN::Vector<NeoN::scalar>& x,
    const NeoN::Vector<NeoN::scalar>& vol
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
    std::vector<NeoN::scalar> r(static_cast<std::size_t>(xv.size()));
    for (NeoN::localIdx i = 0; i < xv.size(); ++i)
    {
        NeoN::scalar ax = 0;
        for (auto j = rows[i]; j < rows[i + 1]; ++j) ax += vals[j] * xv[cols[j]];
        r[static_cast<std::size_t>(i)] = (ax - rhs[i]) / v[i];
    }
    return r;
}

// Per-volume diagonal and summed off-diagonal magnitude of an assembled scalar system (the
// diagonal carries the implicit boundary contributions, as OpenFOAM's fvMatrix::D() does).
template<typename LinearSystem>
std::pair<std::vector<NeoN::scalar>, std::vector<NeoN::scalar>>
diagAndOffDiag(const LinearSystem& ls, const NeoN::Vector<NeoN::scalar>& vol)
{
    const auto valsH = ls.matrix().values().copyToHost();
    const auto rowsH = ls.matrix().sparsity()->rowOffs().copyToHost();
    const auto colsH = ls.matrix().sparsity()->colIdxs().copyToHost();
    const auto volH = vol.copyToHost();
    const auto [vals, rows, cols, v] = NeoN::views(valsH, rowsH, colsH, volH);
    std::vector<NeoN::scalar> d(static_cast<std::size_t>(v.size()), 0), o(d);
    for (NeoN::localIdx i = 0; i < v.size(); ++i)
    {
        for (auto j = rows[i]; j < rows[i + 1]; ++j)
        {
            if (cols[j] == i) d[static_cast<std::size_t>(i)] += vals[j] / v[i];
            else o[static_cast<std::size_t>(i)] += Foam::mag(vals[j]) / v[i];
        }
    }
    return {d, o};
}

// OpenFOAM's counterpart: D()/V and the summed off-diagonal magnitude per cell over V.
std::pair<Foam::scalarField, Foam::scalarField> diagAndOffDiag(const Foam::fvScalarMatrix& M)
{
    const auto& mesh = M.psi().mesh();
    Foam::scalarField d(M.D() / mesh.V().field());
    Foam::scalarField o(d.size(), 0);
    M.sumMagOffDiag(o);
    o /= mesh.V().field();
    return {d, o};
}

// The slice [start, start + n) of a NeoN boundary vector.
std::vector<NeoN::scalar>
patchSlice(const std::vector<NeoN::scalar>& boundary, NeoN::localIdx start, Foam::label n)
{
    return std::vector<NeoN::scalar>(
        boundary.begin() + start, boundary.begin() + start + static_cast<NeoN::localIdx>(n)
    );
}

nf::KEpsilon::Coefficients caseCoefficients(const Foam::Time& runTime)
{
    const Foam::IOdictionary turbProps(Foam::IOobject(
        "turbulenceProperties",
        runTime.constant(),
        runTime,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ));
    return nf::readKEpsilonCoefficients(turbProps);
}

// Both codes' k-epsilon models on the case fields at the start time, U and phi frozen.
struct Models
{
    Foam::volVectorField U;
    Foam::surfaceScalarField phi;
    Foam::singlePhaseTransportModel transport;
    Foam::tmp<Foam::volScalarField> tnu;
    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb;
    Foam::volScalarField wallDist;
    nf::RunTime& rt;
    fvcc::VolumeField<NeoN::Vec3>& nfU;
    fvcc::SurfaceField<NeoN::scalar>& nfPhi;
    fvcc::VolumeField<NeoN::scalar>& nfK;
    fvcc::VolumeField<NeoN::scalar>& nfEps;
    fvcc::VolumeField<NeoN::scalar> nfWallDist;
    fvcc::VolumeField<NeoN::scalar> nfNu;
    fvcc::VolumeField<NeoN::scalar> nfNut;
    nf::KEpsilon turbNF;

    Models(
        Foam::Time& runTime,
        NeoFOAM::MeshAdapter& mesh,
        nf::RunTime& rtIn,
        fvcc::VectorCollection& fieldCollection
    )
        : U(Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ), mesh),
          phi(
              Foam::IOobject(
                  "phi",
                  runTime.timeName(),
                  mesh,
                  Foam::IOobject::NO_READ,
                  Foam::IOobject::NO_WRITE
              ),
              fvc::flux(U)
          ),
          transport(U, phi), tnu(transport.nu()),
          foamTurb(Foam::incompressible::turbulenceModel::New(U, phi, transport)),
          wallDist((foamTurb->validate(), nf::makeNearWallDistField(mesh))), rt(rtIn),
          nfU(NeoFOAM::constructAndRegister(fieldCollection, rt, U, false)),
          nfPhi(NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false)),
          nfK(NeoFOAM::constructAndRegister(
              fieldCollection, rt, mesh.lookupObject<Foam::volScalarField>("k"), false
          )),
          nfEps(NeoFOAM::constructAndRegister(
              fieldCollection, rt, mesh.lookupObject<Foam::volScalarField>("epsilon"), false
          )),
          nfWallDist(nf::constructFrom(rt.exec, rt.nfMesh, wallDist)),
          nfNu(nf::constructFrom(rt.exec, rt.nfMesh, tnu())),
          nfNut(nf::constructFrom(
              rt.exec, rt.nfMesh, mesh.lookupObject<Foam::volScalarField>("nut")
          )),
          turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist)
    {
        turbNF.setCoefficients(caseCoefficients(runTime));
        turbNF.setWallRoughness(NeoN::Vector<NeoN::scalar>(
            rt.exec,
            nf::readEpsilonWallRoughness(
                mesh.lookupObject<Foam::volScalarField>("epsilon"), rt.nfMesh
            )
        ));
        turbNF.validate(nfU, nfK, nfEps, nfNut, &nfPhi);
    }

    void correct()
    {
        turbNF.correct(nfU, nfPhi, nfK, nfEps, nfNut, rt);
        foamTurb->correct();
    }
};

nf::RunTime prepareRunTime(Foam::Time& runTime, const NeoN::Executor& exec)
{
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("k") = nf::mapFvSolution(solverDict.subDict("k"));
    solverDict.subDict("epsilon") = nf::mapFvSolution(solverDict.subDict("epsilon"));
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
    solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);
    return rt;
}
} // namespace

TEST_CASE("kEpsilon: wall-function inputs at a Buildings wall cell match OpenFOAM")
{
    Foam::Time& runTime = *timePtr;
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = prepareRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    REQUIRE(mesh.nCells() == 428);

    Models m(runTime, mesh, rt, fieldCollection);

    const Foam::label patchi = mesh.boundaryMesh().findPatchID("Buildings");
    const auto& patch = mesh.boundary()[patchi];
    const auto start = m.nfK.boundaryData().range(static_cast<NeoN::localIdx>(patchi)).first;

    // The Buildings faces of the hot cell, and the other cells to report.
    std::vector<Foam::label> hotFaces;
    forAll(patch.faceCells(), f)
        if (patch.faceCells()[f] == hotCell) hotFaces.push_back(f);
    REQUIRE(hotFaces.size() == 1);

    std::cout << "=== wall-function inputs on Buildings (" << execName << ")\n";

    const auto relY = compare(
        "y",
        patchSlice(toStd(m.nfWallDist.boundaryData().value()), start, patch.size()),
        m.foamTurb->y()[patchi],
        hotFaces
    );
    const auto relDelta = compare(
        "deltaCoeffs",
        patchSlice(toStd(rt.nfMesh.boundaryMesh().deltaCoeffs()), start, patch.size()),
        patch.deltaCoeffs(),
        hotFaces
    );

    // |dU/dn| at the wall: OpenFOAM's U snGrad against the same difference on NeoN's geometry.
    std::vector<NeoN::scalar> nfMagGradUw(static_cast<std::size_t>(patch.size()));
    {
        const auto uIn = m.nfU.internalVector().copyToHost();
        const auto uB = m.nfU.boundaryData().value().copyToHost();
        const auto own = rt.nfMesh.boundaryMesh().faceOwners().copyToHost();
        const auto dc = rt.nfMesh.boundaryMesh().deltaCoeffs().copyToHost();
        forAll(patch, f)
        {
            const auto i = start + static_cast<NeoN::localIdx>(f);
            nfMagGradUw[static_cast<std::size_t>(f)] =
                NeoN::mag((uB.view()[i] - uIn.view()[own.view()[i]]) * dc.view()[i]);
        }
    }
    const auto relGradUw = compare(
        "|dU/dn| wall", nfMagGradUw, Foam::mag(m.U.boundaryField()[patchi].snGrad())(), hotFaces
    );

    const auto& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");
    const auto relNutW = compare(
        "nut_w",
        patchSlice(toStd(m.nfNut.boundaryData().value()), start, patch.size()),
        ofNut.boundaryField()[patchi],
        hotFaces
    );

    // The production of one step. OpenFOAM's G: nut GbyNu0 in the bulk, overwritten in the wall
    // cells by its epsilonWallFunction (updateCoeffs looks the field up by name). NeoFOAM's Pk_
    // additionally carries the explicit half of the k dilatation sink, -min((2/3) divU, 0) k.
    Foam::volScalarField::Internal G(
        Foam::IOobject(
            m.foamTurb->GName(),
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::REGISTER
        ),
        ofNut()
            * (Foam::dev(Foam::twoSymm(fvc::grad(m.U)().v())) && fvc::grad(m.U)().v())
    );
    const auto& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::scalarField divU(fvc::div(m.phi)().primitiveField());
    const Foam::scalarField kStart(ofK.primitiveField());
    const_cast<Foam::volScalarField&>(mesh.lookupObject<Foam::volScalarField>("epsilon"))
        .boundaryFieldRef()
        .updateCoeffs();
    const Foam::scalarField ofPk(G.field() - Foam::min((2.0 / 3.0) * divU, 0.0) * kStart);

    m.turbNF.correct(m.nfU, m.nfPhi, m.nfK, m.nfEps, m.nfNut, rt);
    const auto relPk = compare(
        "production (G + dilatation)",
        toStd(m.turbNF.pkField().internalVector()),
        ofPk,
        {hotCell}
    );

    CHECK(relY < 1e-8);
    CHECK(relDelta < 1e-8);
    CHECK(relGradUw < 1e-8);
    CHECK(relNutW < 1e-8);
    CHECK(relPk < 1e-8);
}

TEST_CASE("kEpsilon: one turbulence step at a Buildings wall cell matches OpenFOAM")
{
    Foam::Time& runTime = *timePtr;
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = prepareRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    Models m(runTime, mesh, rt, fieldCollection);
    const auto& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::scalarField kStart(ofK.primitiveField());
    std::vector<bool> isWall(static_cast<std::size_t>(mesh.nCells()), false);
    for (const auto c : mesh.boundary()[mesh.boundaryMesh().findPatchID("Buildings")].faceCells())
        isWall[static_cast<std::size_t>(c)] = true;
    m.correct();

    const auto& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const auto& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    std::cout << "=== one turbulence step (" << execName << ")\n";
    const auto relK = compare("k", toStd(m.nfK.internalVector()), ofK, {hotCell});
    const auto relEps = compare("epsilon", toStd(m.nfEps.internalVector()), ofEps, {hotCell});
    const auto relNut = compare("nut", toStd(m.nfNut.internalVector()), ofNut, {hotCell});

    // The cells where k differs most, with their start value.
    {
        const auto k = toStd(m.nfK.internalVector());
        std::vector<Foam::label> order(static_cast<std::size_t>(mesh.nCells()));
        std::iota(order.begin(), order.end(), 0);
        std::sort(
            order.begin(),
            order.end(),
            [&](auto a, auto b)
            {
                return Foam::mag(k[static_cast<std::size_t>(a)] - ofK[a])
                     > Foam::mag(k[static_cast<std::size_t>(b)] - ofK[b]);
            }
        );
        std::cout << "  worst cells: cell  k start  neo  of  wall\n";
        for (std::size_t r = 0; r < 10; ++r)
        {
            const auto c = order[r];
            std::cout << "  " << c << "  " << kStart[c] << "  " << k[static_cast<std::size_t>(c)]
                      << "  " << ofK[c] << "  " << isWall[static_cast<std::size_t>(c)] << "\n";
        }
    }

    CHECK(relK < 1e-6);
    CHECK(relEps < 1e-6);
    CHECK(relNut < 1e-6);
}

TEST_CASE("kEpsilon: k and epsilon transport terms at a Buildings wall cell match OpenFOAM")
{
    Foam::Time& runTime = *timePtr;
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = prepareRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    Models m(runTime, mesh, rt, fieldCollection);
    const auto& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const auto& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const auto& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    // (M & epsilon) updates epsilon's wall-function patches, which look up the model's
    // production field; kEpsilon only creates it inside correct().
    Foam::volScalarField::Internal G(
        Foam::IOobject(
            m.foamTurb->GName(),
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::REGISTER
        ),
        mesh,
        Foam::dimensionedScalar(Foam::sqr(Foam::dimVelocity) / Foam::dimTime, Foam::Zero)
    );

    const auto coeffs = m.turbNF.coeffs();
    const Foam::volScalarField ofDkEff("DkEff", ofNut / coeffs.sigmaK + m.tnu());
    const Foam::volScalarField ofDepsEff("DepsilonEff", ofNut / coeffs.sigmaEps + m.tnu());

    const auto& vol = rt.nfMesh.cellVolumes();
    auto residualOf = [&](auto& field, auto&& expr)
    {
        nf::PDE<NeoN::scalar> pde(NeoN::dsl::Expression<NeoN::scalar>(expr), field, rt);
        pde.assemble();
        return cellResidual(pde.linearSystem(), field.internalVector(), vol);
    };
    namespace dsl = NeoN::dsl;
    namespace fvm = Foam::fvm;

    // Building an epsilon matrix runs the epsilonWallFunction's updateCoeffs(), which sets the
    // wall cells and faces to epsilon0. Do it once up front and compare both codes on that state.
    const_cast<Foam::volScalarField&>(ofEps).boundaryFieldRef().updateCoeffs();
    auto nfEpsW = nf::constructFrom(rt.exec, rt.nfMesh, ofEps);

    std::cout << "=== k and epsilon transport terms (" << execName << ")\n";
    const std::vector<Foam::label> cells {hotCell, 31};
    const auto relDivK = compare(
        "div(phi,k)",
        residualOf(m.nfK, dsl::imp::div(m.nfPhi, m.nfK)),
        (Foam::fvScalarMatrix(fvm::div(m.phi, ofK)) & ofK)().primitiveField(),
        cells
    );
    const auto relLapK = compare(
        "laplacian(DkEff,k)",
        residualOf(m.nfK, dsl::imp::laplacian(m.turbNF.dkEff(), m.nfK)),
        (Foam::fvScalarMatrix(fvm::laplacian(ofDkEff, ofK)) & ofK)().primitiveField(),
        cells
    );
    const auto relDivEps = compare(
        "div(phi,epsilon)",
        residualOf(nfEpsW, dsl::imp::div(m.nfPhi, nfEpsW)),
        (Foam::fvScalarMatrix(fvm::div(m.phi, ofEps)) & ofEps)().primitiveField(),
        cells
    );
    const auto relLapEps = compare(
        "laplacian(DepsilonEff,epsilon)",
        residualOf(nfEpsW, dsl::imp::laplacian(m.turbNF.depsilonEff(), nfEpsW)),
        (Foam::fvScalarMatrix(fvm::laplacian(ofDepsEff, ofEps)) & ofEps)().primitiveField(),
        cells
    );

    // The implicit/explicit split: same residual, but is the same part on the diagonal?
    NeoN::scalar relDiagDivK = 0, relDiagLapK = 0;
    {
        auto split = [&](auto&& expr)
        {
            nf::PDE<NeoN::scalar> pde(NeoN::dsl::Expression<NeoN::scalar>(expr), m.nfK, rt);
            pde.assemble();
            return diagAndOffDiag(pde.linearSystem(), vol);
        };
        const auto [nDiv, nDivO] = split(dsl::imp::div(m.nfPhi, m.nfK));
        const auto [oDiv, oDivO] = diagAndOffDiag(Foam::fvScalarMatrix(fvm::div(m.phi, ofK)));
        relDiagDivK = compare("div(phi,k) diagonal", nDiv, oDiv, cells);
        compare("div(phi,k) sum|offdiag|", nDivO, oDivO, cells);
        const auto [nLap, nLapO] = split(dsl::imp::laplacian(m.turbNF.dkEff(), m.nfK));
        const auto [oLap, oLapO] =
            diagAndOffDiag(Foam::fvScalarMatrix(fvm::laplacian(ofDkEff, ofK)));
        relDiagLapK = compare("laplacian(DkEff,k) diagonal", nLap, oLap, cells);
        compare("laplacian(DkEff,k) sum|offdiag|", nLapO, oLapO, cells);
    }

    CHECK(relDivK < 1e-8);
    CHECK(relLapK < 1e-8);
    CHECK(relDiagDivK < 1e-8);
    CHECK(relDiagLapK < 1e-8);
    CHECK(relDivEps < 1e-8);
    CHECK(relLapEps < 1e-8);
}

TEST_CASE("one SIMPLE iteration at a Buildings wall region matches OpenFOAM stage by stage")
{
    Foam::Time& runTime = *timePtr;
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = prepareRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    Models m(runTime, mesh, rt, fieldCollection);
    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ), mesh
    );
    mesh.setFluxRequired(p.name());
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);

    const Foam::label nNonOrth = mesh.solutionDict().subDict("SIMPLE").getOrDefault<Foam::label>(
        "nNonOrthogonalCorrectors", 0
    );
    const Foam::label nIntFaces = mesh.nInternalFaces();
    const std::vector<Foam::label> cells {hotCell, 31};
    namespace dsl = NeoN::dsl;

    // ---- NeoFOAM: the body of neoSimpleFoam's SIMPLE loop ----
    std::vector<NeoN::Vec3> nUStar, nHbyA, nUCorr;
    std::vector<NeoN::scalar> nRAU, nPhiHbyA, nPSolved, nPhi, nPRelaxed, nK, nEps, nNut;
    {
        auto& U = m.nfU;
        auto& phi = m.nfPhi;
        fvcc::BoundaryContext uCtx;
        uCtx.insert("phi", phi);
        U.correctBoundaryConditions(uCtx);

        nf::PDESolver<NeoN::Vec3> UEqn(
            dsl::imp::div(phi, U) - dsl::imp::laplacian(m.turbNF.nuEff(), U)
                + dsl::exp::viscousStress(m.nfNu, m.nfNut, m.turbNF.gradU()),
            U,
            rt
        );
        UEqn.relax();
        UEqn.solve(-1.0 * dsl::exp::grad(nfP));
        nUStar = toStd(U.internalVector());

        auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
        nf::constrainHbyA(U, nfP, hByA);
        nRAU = toStd(crAU.internalVector());
        nHbyA = toStd(hByA.internalVector());

        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec, rt.nfMesh, NeoN::TokenList({std::string("linear")})
        );
        fvcc::SurfaceField<NeoN::scalar> rAUf = surfInterpol.interpolate(crAU);
        rAUf.name = "rAUf";
        auto phiHbyA = nf::flux(hByA);
        nPhiHbyA = head(toStd(phiHbyA.internalVector()), nIntFaces);

        auto pPrev = NeoN::dsl::fieldRelaxationSnapshot(nfP);
        fvcc::BoundaryContext pCtx;
        pCtx.insert("phi", phi);
        pCtx.insert("U", U);
        for (Foam::label corr = 0; corr <= nNonOrth; ++corr)
        {
            nfP.correctBoundaryConditions(pCtx);
            nf::PDESolver<NeoN::scalar> pEqn(
                dsl::imp::laplacian(rAUf, nfP) - dsl::exp::div(phiHbyA), nfP, rt
            );
            pEqn.linearSystem().keepFaceFluxCorrection(true);
            pEqn.solve();
            nfP.correctBoundaryConditions(pCtx);
            if (corr == nNonOrth) nf::updateFaceVelocity(phiHbyA, pEqn, phi);
        }
        nPSolved = toStd(nfP.internalVector());
        nPhi = head(toStd(phi.internalVector()), nIntFaces);

        NeoN::dsl::applyFieldRelaxation(
            nfP,
            pPrev,
            nf::lookupFieldRelaxation(rt.fvSolutionDict, "p", false).value_or(NeoN::scalar(1))
        );
        nfP.correctBoundaryConditions(pCtx);
        nPRelaxed = toStd(nfP.internalVector());

        nf::updateVelocity(hByA, crAU, nfP, U, rt);
        U.correctBoundaryConditions(pCtx);
        nUCorr = toStd(U.internalVector());

        m.turbNF.correct(U, phi, m.nfK, m.nfEps, m.nfNut, rt);
        nK = toStd(m.nfK.internalVector());
        nEps = toStd(m.nfEps.internalVector());
        nNut = toStd(m.nfNut.internalVector());
    }

    // ---- OpenFOAM: simpleFoam's UEqn.H and pEqn.H ----
    auto& U = m.U;
    auto& phi = m.phi;
    p.storePrevIter();
    Foam::tmp<Foam::fvVectorMatrix> tUEqn(
        Foam::fvm::div(phi, U) + m.foamTurb->divDevReff(U)
    );
    Foam::fvVectorMatrix& UEqn = tUEqn.ref();
    UEqn.relax();
    Foam::solve(UEqn == -fvc::grad(p));
    const Foam::vectorField oUStar(U.primitiveField());

    Foam::volScalarField rAU(1.0 / UEqn.A());
    Foam::volVectorField HbyA(Foam::constrainHbyA(rAU * UEqn.H(), U, p));
    Foam::surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
    Foam::adjustPhi(phiHbyA, U, p);
    tUEqn.clear();
    Foam::constrainPressure(p, U, phiHbyA, rAU);
    for (Foam::label corr = 0; corr <= nNonOrth; ++corr)
    {
        Foam::fvScalarMatrix pEqn(Foam::fvm::laplacian(rAU, p) == fvc::div(phiHbyA));
        pEqn.solve();
        if (corr == nNonOrth) phi = phiHbyA - pEqn.flux();
    }
    const Foam::scalarField oPSolved(p.primitiveField());
    p.relax();
    U = HbyA - rAU * fvc::grad(p);
    U.correctBoundaryConditions();
    m.foamTurb->correct();

    const auto& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const auto& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const auto& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    std::cout << "=== one SIMPLE iteration (" << execName << ")\n";
    const auto relUStar = compare("U after the momentum predictor", nUStar, oUStar, cells);
    const auto relRAU = compare("rAU", nRAU, rAU.primitiveField(), cells);
    const auto relHbyA = compare("HbyA", nHbyA, HbyA.primitiveField(), cells);
    const auto relPhiHbyA = compare("phiHbyA", nPhiHbyA, phiHbyA.primitiveField(), {});
    const auto relP = compare("p after the pressure solve", nPSolved, oPSolved, cells);
    const auto relPhi = compare("phi", nPhi, phi.primitiveField(), {});
    const auto relPRelaxed = compare("p relaxed", nPRelaxed, p.primitiveField(), cells);
    const auto relU = compare("U corrected", nUCorr, U.primitiveField(), cells);
    const auto relK = compare("k", nK, ofK.primitiveField(), cells);
    const auto relEps = compare("epsilon", nEps, ofEps.primitiveField(), cells);
    const auto relNut = compare("nut", nNut, ofNut.primitiveField(), cells);

    CHECK(relUStar < 1e-6);
    CHECK(relRAU < 1e-6);
    CHECK(relHbyA < 1e-6);
    CHECK(relPhiHbyA < 1e-6);
    CHECK(relP < 1e-6);
    CHECK(relPhi < 1e-6);
    CHECK(relPRelaxed < 1e-6);
    CHECK(relU < 1e-6);
    CHECK(relK < 1e-6);
    CHECK(relEps < 1e-6);
    CHECK(relNut < 1e-6);
}

// Diagnostic for a full case, not run by ctest ([.] hides it): run the binary inside a case
// directory with "[fullmesh]". Assembles the momentum equation of the first SIMPLE iteration in
// both codes on the case's state -- NeoFOAM as neoSimpleFoam, OpenFOAM as simpleFoam -- and
// compares the per-cell residual A U - b (relaxation does not change it at the current U), summed
// as |r| V by region: the cells next to each patch, and the interior.
TEST_CASE("full-mesh momentum residual by region", "[.fullmesh]")
{
    Foam::Time& runTime = *timePtr;
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");
    NeoN::Executor exec = NeoN::GPUExecutor {};

    auto rt = prepareRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    std::cout << "=== full-mesh momentum residual, " << mesh.nCells() << " cells\n";

    Models m(runTime, mesh, rt, fieldCollection);
    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ), mesh
    );
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    namespace dsl = NeoN::dsl;

    // NeoFOAM: neoSimpleFoam's UEqn with -grad(p) moved to the left.
    std::vector<NeoN::Vec3> nRes;
    {
        fvcc::BoundaryContext uCtx;
        uCtx.insert("phi", m.nfPhi);
        m.nfU.correctBoundaryConditions(uCtx);
        // grad(U) on the flux-split boundary, as OpenFOAM evaluates it in divDevReff.
        m.turbNF.updateGradU(m.nfU);
        nf::PDE<NeoN::Vec3> UEqn(
            dsl::Expression<NeoN::Vec3>(
                dsl::imp::div(m.nfPhi, m.nfU) - dsl::imp::laplacian(m.turbNF.nuEff(), m.nfU)
                + dsl::exp::viscousStress(m.nfNu, m.nfNut, m.turbNF.gradU())
                + dsl::exp::grad(nfP)
            ),
            m.nfU,
            rt
        );
        UEqn.assemble();
        const auto& ls = UEqn.linearSystem();
        const auto valsH = ls.matrix().values().copyToHost();
        const auto rowsH = ls.matrix().sparsity()->rowOffs().copyToHost();
        const auto colsH = ls.matrix().sparsity()->colIdxs().copyToHost();
        const auto rhsH = ls.rhs().copyToHost();
        const auto xH = m.nfU.internalVector().copyToHost();
        const auto volH = rt.nfMesh.cellVolumes().copyToHost();
        const auto [vals, rows, cols, rhs, xv, v] =
            NeoN::views(valsH, rowsH, colsH, rhsH, xH, volH);
        nRes.resize(static_cast<std::size_t>(xv.size()));
        for (NeoN::localIdx i = 0; i < xv.size(); ++i)
        {
            NeoN::Vec3 ax(0, 0, 0);
            for (auto j = rows[i]; j < rows[i + 1]; ++j) ax += vals[j] * xv[cols[j]];
            nRes[static_cast<std::size_t>(i)] = (ax - rhs[i]) * (1.0 / v[i]);
        }
        // The implicit symmetry/slip normal damping lives outside the matrix (diagCmpt) and is
        // subtracted from the diagonal per component at the solve; include it here too.
        if (ls.diagCmpt() && ls.diagCmpt()->size() > 0)
        {
            const auto dcH = ls.diagCmpt()->copyToHost();
            const auto dc = dcH.view();
            for (NeoN::localIdx i = 0; i < xv.size(); ++i)
            {
                for (int c = 0; c < 3; ++c)
                {
                    nRes[static_cast<std::size_t>(i)][c] -= dc[i][c] * xv[i][c] / v[i];
                }
            }
            std::cout << "diagCmpt present (implicit symmetry/slip damping)\n";
        }
    }

    // The boundary values the explicit stress uses, patch by patch: nut, U and grad(U).
    {
        const auto& bmP = mesh.boundaryMesh();
        const auto nutB = toStd(m.nfNut.boundaryData().value());
        const auto uB = toStd(m.nfU.boundaryData().value());
        const auto gH = m.turbNF.gradU().boundaryData().value().copyToHost();
        const auto& ofNutB = mesh.lookupObject<Foam::volScalarField>("nut").boundaryField();
        const Foam::volTensorField ofGrad(fvc::grad(m.U));
        std::cout << "boundary values: patch  max|d nut|/max|nut|  max|d U|/max|U|  "
                     "max|d gradU|/max|gradU|\n";
        forAll(bmP, patchi)
        {
            if (bmP[patchi].empty()) continue;
            const auto start = m.nfNut.boundaryData().range(static_cast<NeoN::localIdx>(patchi)).first;
            NeoN::scalar dn = 0, mn = 0, du = 0, mu = 0, dg = 0, mg = 0;
            forAll(bmP[patchi], f)
            {
                const auto i = static_cast<std::size_t>(start + f);
                dn = Foam::max(dn, Foam::mag(nutB[i] - ofNutB[patchi][f]));
                mn = Foam::max(mn, Foam::mag(ofNutB[patchi][f]));
                const Foam::vector u(uB[i][0], uB[i][1], uB[i][2]);
                du = Foam::max(du, Foam::mag(u - m.U.boundaryField()[patchi][f]));
                mu = Foam::max(mu, Foam::mag(m.U.boundaryField()[patchi][f]));
                // NeoN stores dU_i/dx_j at (i, j), OpenFOAM at (j, i).
                const auto& g = gH.view()[static_cast<NeoN::localIdx>(i)];
                const Foam::tensor gT(
                    g(0, 0), g(1, 0), g(2, 0), g(0, 1), g(1, 1), g(2, 1), g(0, 2), g(1, 2), g(2, 2)
                );
                dg = Foam::max(dg, Foam::mag(gT - ofGrad.boundaryField()[patchi][f]));
                mg = Foam::max(mg, Foam::mag(ofGrad.boundaryField()[patchi][f]));
            }
            std::cout << "  " << bmP[patchi].name() << "  " << dn / Foam::max(mn, 1e-300) << "  "
                      << du / Foam::max(mu, 1e-300) << "  " << dg / Foam::max(mg, 1e-300) << "\n";
        }
        // The geometry of the boundary correction, and the boundary gradient row by row.
        const auto dcN = toStd(rt.nfMesh.boundaryMesh().deltaCoeffs());
        const auto nN = toStd(rt.nfMesh.boundaryMesh().faceUnitNormals());
        std::cout << "boundary geometry: patch  max|d deltaCoeffs|/max  max|d n|  "
                     "worst grad(U) face: neo / of (OpenFOAM layout)\n";
        forAll(bmP, patchi)
        {
            if (bmP[patchi].empty()) continue;
            const auto start = m.nfNut.boundaryData().range(static_cast<NeoN::localIdx>(patchi)).first;
            const auto& dcO = mesh.boundary()[patchi].deltaCoeffs();
            const auto nO = mesh.boundary()[patchi].nf();
            NeoN::scalar dd = 0, md = 0, dnn = 0, dg = 0;
            Foam::label worst = -1;
            forAll(bmP[patchi], f)
            {
                const auto i = static_cast<std::size_t>(start + f);
                dd = Foam::max(dd, Foam::mag(dcN[i] - dcO[f]));
                md = Foam::max(md, dcO[f]);
                dnn = Foam::max(
                    dnn, Foam::mag(Foam::vector(nN[i][0], nN[i][1], nN[i][2]) - nO()[f])
                );
                const auto& g = gH.view()[static_cast<NeoN::localIdx>(i)];
                const Foam::tensor gT(
                    g(0, 0), g(1, 0), g(2, 0), g(0, 1), g(1, 1), g(2, 1), g(0, 2), g(1, 2), g(2, 2)
                );
                const NeoN::scalar d = Foam::mag(gT - ofGrad.boundaryField()[patchi][f]);
                if (d > dg)
                {
                    dg = d;
                    worst = f;
                }
            }
            std::cout << "  " << bmP[patchi].name() << "  " << dd / Foam::max(md, 1e-300) << "  "
                      << dnn;
            if (worst >= 0 && dg > 1e-8)
            {
                const auto i = static_cast<NeoN::localIdx>(start + worst);
                const auto& g = gH.view()[i];
                std::cout << "  face " << worst << " cell " << bmP[patchi].faceCells()[worst]
                          << ": neo (" << g(0, 0) << " " << g(1, 0) << " " << g(2, 0) << " / "
                          << g(0, 1) << " " << g(1, 1) << " " << g(2, 1) << " / " << g(0, 2) << " "
                          << g(1, 2) << " " << g(2, 2) << ") of (";
                const auto& o = ofGrad.boundaryField()[patchi][worst];
                for (int k = 0; k < 9; ++k) std::cout << o[k] << (k % 3 == 2 && k < 8 ? " / " : " ");
                const auto& ub = m.U.boundaryField()[patchi][worst];
                const auto& uc = m.U[bmP[patchi].faceCells()[worst]];
                std::cout << ") U_b (" << ub.x() << " " << ub.y() << " " << ub.z() << ") U_c ("
                          << uc.x() << " " << uc.y() << " " << uc.z() << ")";
            }
            std::cout << "\n";
        }
        // The cell gradient of the cells next to each patch.
        const auto gI = m.turbNF.gradU().internalVector().copyToHost();
        std::cout << "cell grad(U) next to: patch  max|d|/max|gradU|  worst cell\n";
        forAll(bmP, patchi)
        {
            if (bmP[patchi].empty()) continue;
            NeoN::scalar dg = 0, mg = 0;
            Foam::label at = -1;
            for (const auto c : bmP[patchi].faceCells())
            {
                const auto& g = gI.view()[c];
                const Foam::tensor gT(
                    g(0, 0), g(1, 0), g(2, 0), g(0, 1), g(1, 1), g(2, 1), g(0, 2), g(1, 2), g(2, 2)
                );
                const NeoN::scalar d = Foam::mag(gT - ofGrad[c]);
                if (d > dg)
                {
                    dg = d;
                    at = c;
                }
                mg = Foam::max(mg, Foam::mag(ofGrad[c]));
            }
            std::cout << "  " << bmP[patchi].name() << "  " << dg / Foam::max(mg, 1e-300) << "  "
                      << at << "\n";
        }
    }

    // OpenFOAM: simpleFoam's UEqn, residual (M & U) + grad(p).
    Foam::fvVectorMatrix UEqn(Foam::fvm::div(m.phi, m.U) + m.foamTurb->divDevReff(m.U));
    const Foam::vectorField oRes(((UEqn & m.U)().primitiveField()) + fvc::grad(p)().primitiveField());

    // Region of each cell: the first patch it touches, else interior.
    const auto& bm = mesh.boundaryMesh();
    Foam::labelList region(mesh.nCells(), bm.size());
    forAll(bm, patchi)
    {
        for (const auto c : bm[patchi].faceCells())
            if (region[c] == bm.size()) region[c] = patchi;
    }
    const auto& V = mesh.V();
    const auto& C = mesh.C();
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        std::vector<NeoN::scalar> sN(bm.size() + 1, 0), sO(bm.size() + 1, 0), sD(bm.size() + 1, 0);
        std::vector<Foam::label> n(bm.size() + 1, 0);
        forAll(oRes, c)
        {
            const auto r = static_cast<std::size_t>(region[c]);
            const NeoN::scalar rn = nRes[static_cast<std::size_t>(c)][cmpt], ro = oRes[c][cmpt];
            sN[r] += Foam::mag(rn) * V[c];
            sO[r] += Foam::mag(ro) * V[c];
            sD[r] += Foam::mag(rn - ro) * V[c];
            ++n[r];
        }
        NeoN::scalar tN = 0, tO = 0, tD = 0;
        for (std::size_t r = 0; r < sN.size(); ++r)
        {
            tN += sN[r];
            tO += sO[r];
            tD += sD[r];
        }
        std::cout << "component " << "xyz"[cmpt] << ": sum|r|V neo " << tN << " of " << tO
                  << ", sum|r_neo - r_of|V " << tD << "\n";
        for (std::size_t r = 0; r < sN.size(); ++r)
        {
            const std::string name =
                r < static_cast<std::size_t>(bm.size()) ? std::string(bm[r].name()) : "interior";
            std::cout << "  " << name << " (" << n[r] << " cells): neo " << sN[r] << " of " << sO[r]
                      << " |d| " << sD[r] << " (" << 100 * sD[r] / Foam::max(tD, 1e-300)
                      << "% of the difference)\n";
        }
    }

    // The difference by term, in the regions next to each patch.
    {
        auto nfTerm = [&](auto&& expr)
        {
            nf::PDE<NeoN::Vec3> pde(dsl::Expression<NeoN::Vec3>(expr), m.nfU, rt);
            pde.assemble();
            const auto& ls = pde.linearSystem();
            auto r = cellResidualVec(ls, m.nfU.internalVector(), rt.nfMesh.cellVolumes());
            if (ls.diagCmpt() && ls.diagCmpt()->size() > 0)
            {
                const auto dcH = ls.diagCmpt()->copyToHost();
                const auto xH = m.nfU.internalVector().copyToHost();
                for (Foam::label c = 0; c < mesh.nCells(); ++c)
                    for (int k = 0; k < 3; ++k)
                        r[static_cast<std::size_t>(c)][k] -=
                            dcH.view()[c][k] * xH.view()[c][k] / mesh.V()[c];
            }
            return r;
        };
        const Foam::volScalarField nuEff(m.foamTurb->nuEff());
        struct Term
        {
            std::string name;
            std::vector<NeoN::Vec3> nfR;
            Foam::vectorField ofR;
        };
        std::vector<Term> terms;
        terms.push_back(
            {"div(phi,U)",
             nfTerm(dsl::imp::div(m.nfPhi, m.nfU)),
             (Foam::fvVectorMatrix(Foam::fvm::div(m.phi, m.U)) & m.U)().primitiveField()}
        );
        terms.push_back(
            {"-laplacian(nuEff,U)",
             nfTerm(-1.0 * dsl::imp::laplacian(m.turbNF.nuEff(), m.nfU)),
             (Foam::fvVectorMatrix(-Foam::fvm::laplacian(nuEff, m.U)) & m.U)().primitiveField()}
        );
        terms.push_back(
            {"viscousStress",
             nfTerm(dsl::exp::viscousStress(m.nfNu, m.nfNut, m.turbNF.gradU())),
             (-fvc::div(nuEff * Foam::dev2(Foam::T(fvc::grad(m.U)))))().primitiveField()}
        );
        terms.push_back({"grad(p)", nfTerm(dsl::exp::grad(nfP)), fvc::grad(p)().primitiveField()});
        for (const auto& t : terms)
        {
            std::vector<NeoN::scalar> d(bm.size() + 1, 0), o(bm.size() + 1, 0), nn(bm.size() + 1, 0);
            forAll(oRes, c)
            {
                const auto r = static_cast<std::size_t>(region[c]);
                d[r] += Foam::mag(t.nfR[static_cast<std::size_t>(c)][2] - t.ofR[c][2]) * V[c];
                o[r] += Foam::mag(t.ofR[c][2]) * V[c];
                nn[r] += Foam::mag(t.nfR[static_cast<std::size_t>(c)][2]) * V[c];
            }
            std::cout << "term " << t.name << " (z): sum|r_neo - r_of|V by region:";
            for (std::size_t r = 0; r < d.size(); ++r)
            {
                if (d[r] < 1e-6 * Foam::max(o[r], 1e-300) && d[r] < 1e-3) continue;
                const std::string name = r < static_cast<std::size_t>(bm.size())
                                           ? std::string(bm[r].name())
                                           : "interior";
                std::cout << " " << name << " " << d[r] << " (neo " << nn[r] << ", of " << o[r]
                          << ")";
            }
            std::cout << "\n";
        }
    }

    // The cells where the codes differ most in z.
    std::vector<Foam::label> order(static_cast<std::size_t>(mesh.nCells()));
    std::iota(order.begin(), order.end(), 0);
    auto dz = [&](Foam::label c)
    { return Foam::mag(nRes[static_cast<std::size_t>(c)][2] - oRes[c][2]) * V[c]; };
    std::partial_sort(
        order.begin(), order.begin() + 20, order.end(), [&](auto a, auto b) { return dz(a) > dz(b); }
    );
    std::cout << "worst cells in z: cell region centre r_neo r_of (per volume) V\n";
    for (int i = 0; i < 20; ++i)
    {
        const auto c = order[static_cast<std::size_t>(i)];
        const std::string name = region[c] < bm.size() ? std::string(bm[region[c]].name())
                                                      : "interior";
        std::cout << "  " << c << " " << name << " (" << C[c].x() << " " << C[c].y() << " "
                  << C[c].z() << ") " << nRes[static_cast<std::size_t>(c)][2] << " "
                  << oRes[c][2] << " " << V[c] << "\n";
    }
}
