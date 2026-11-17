// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <numeric>
#include <vector>

#include "NeoN/NeoN.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/auxiliary/readers.hpp"
#include "detail.hpp"
#include "foamGeometry.hpp"

#include "surfaceFields.H"
#include "volFields.H"

namespace NeoFOAM
{

using NeoN::localIdx;
using NeoN::scalar;

const FoamGeometry& foamGeometry(RunTime& rt)
{
    auto& db = rt.nfMesh.stencilDB();
    if (!db.contains("FoamGeometry"))
    {
        const Foam::vectorField& centres = rt.mesh.C();
        const Foam::labelUList& owner = rt.mesh.owner();
        const Foam::labelUList& neighbour = rt.mesh.neighbour();
        Foam::vectorField delta(owner.size());
        forAll(delta, f)
        {
            delta[f] = centres[neighbour[f]] - centres[owner[f]];
        }
        db.insert(
            "FoamGeometry",
            std::make_shared<FoamGeometry>(FoamGeometry {
                fromFoamField(rt.exec, rt.mesh.weights().primitiveField()),
                fromFoamField(rt.exec, delta),
                fromFoamField(rt.exec, rt.mesh.nonOrthDeltaCoeffs().primitiveField()),
                fromFoamField(rt.exec, rt.mesh.nonOrthCorrectionVectors().primitiveField())
            })
        );
    }
    return *db.get<std::shared_ptr<FoamGeometry>>("FoamGeometry");
}

const CellFaces& cellFaces(const NeoN::UnstructuredMesh& mesh)
{
    auto& db = mesh.stencilDB();
    if (!db.contains("VofCellFaces"))
    {
        const auto ownHost = mesh.faceOwners().copyToHost();
        const auto neiHost = mesh.faceNeighbors().copyToHost();
        const auto bOwnHost = mesh.boundaryMesh().faceOwners().copyToHost();
        const auto own = ownHost.view();
        const auto nei = neiHost.view();
        const auto bOwn = bOwnHost.view();
        const localIdx nInternal = mesh.nInternalFaces();
        const auto nBoundary = static_cast<localIdx>(bOwn.size());

        std::vector<localIdx> offsets(static_cast<std::size_t>(mesh.nCells()) + 1, 0);
        for (localIdx f = 0; f < nInternal; ++f)
        {
            ++offsets[own[f] + 1];
            ++offsets[nei[f] + 1];
        }
        for (localIdx bf = 0; bf < nBoundary; ++bf)
        {
            ++offsets[bOwn[bf] + 1];
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

        std::vector<localIdx> next(offsets.begin(), offsets.end() - 1);
        std::vector<localIdx> faces(offsets.back());
        for (localIdx f = 0; f < nInternal; ++f)
        {
            faces[next[own[f]]++] = f;
            faces[next[nei[f]]++] = f;
        }
        for (localIdx bf = 0; bf < nBoundary; ++bf)
        {
            faces[next[bOwn[bf]]++] = nInternal + bf;
        }
        db.insert(
            "VofCellFaces",
            std::make_shared<CellFaces>(CellFaces {
                NeoN::Vector<localIdx>(mesh.exec(), offsets),
                NeoN::Vector<localIdx>(mesh.exec(), faces)
            })
        );
    }
    return *db.get<std::shared_ptr<CellFaces>>("VofCellFaces");
}

NeoN::Vector<NeoN::Vec3>
gaussLinearGrad(const nnfvcc::VolumeField<scalar>& psi, NeoN::View<const scalar> weights)
{
    const auto& mesh = psi.mesh();
    const auto exec = psi.exec();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto sf = mesh.faceNormals().view();
    const auto bSf = mesh.boundaryMesh().faceNormals().view();
    const auto V = mesh.cellVolumes().view();
    const auto psiI = psi.internalVector().view();
    const auto psiB = psi.boundaryData().value().view();

    NeoN::Vector<NeoN::Vec3> grad(exec, mesh.nCells(), NeoN::Vec3(0.0, 0.0, 0.0));
    auto g = grad.view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const NeoN::Vec3 sfPsi = sf[f] * faceValue(weights[f], psiI[own[f]], psiI[nei[f]]);
            for (int i = 0; i < 3; ++i)
            {
                Kokkos::atomic_add(&g[own[f]][i], sfPsi[i]);
                Kokkos::atomic_sub(&g[nei[f]][i], sfPsi[i]);
            }
        },
        "vof::gradInternal"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) {
            const NeoN::Vec3 sfPsi = bSf[bf] * psiB[bf];
            for (int i = 0; i < 3; ++i)
            {
                Kokkos::atomic_add(&g[bOwn[bf]][i], sfPsi[i]);
            }
        },
        "vof::gradBoundary"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { g[c] = g[c] / V[c]; },
        "vof::gradVolume"
    );
    return grad;
}

void integrateFluxInto(
    const NeoN::UnstructuredMesh& mesh,
    NeoN::View<const scalar> flux,
    NeoN::View<const scalar> bFlux,
    NeoN::View<scalar> res
)
{
    const auto exec = mesh.exec();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto V = mesh.cellVolumes().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            Kokkos::atomic_add(&res[own[f]], flux[f]);
            Kokkos::atomic_sub(&res[nei[f]], flux[f]);
        },
        "vof::integrateInternal"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&res[bOwn[bf]], bFlux[bf]); },
        "vof::integrateBoundary"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { res[c] /= V[c]; },
        "vof::integrateVolume"
    );
}

} // namespace NeoFOAM
