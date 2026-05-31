// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"

#include "processorFvPatch.H"
#include "lduInterfaceField.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //


namespace NeoFOAM
{

template<typename FieldT>
FieldT flatBCField(const Foam::fvMesh& mesh, std::function<FieldT(const Foam::fvPatch&)> f)
{
    FieldT result(computeNBoundaryFaces(mesh));
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    Foam::label idx = 0;
    forAll(bMesh, patchI)
    {
        const Foam::fvPatch& patch = bMesh[patchI];
        if (Foam::isA<Foam::processorFvPatch>(patch))
        {
            continue;
        }
        auto pResult = f(patch);
        forAll(pResult, i)
        {
            result[idx] = pResult[i];
            idx++;
        }
    }
    forAll(bMesh, patchI)
    {
        const Foam::fvPatch& patch = bMesh[patchI];
        if (!Foam::isA<Foam::processorFvPatch>(patch))
        {
            continue;
        }
        auto pResult = f(patch);
        forAll(pResult, i)
        {
            result[idx] = pResult[i];
            idx++;
        }
    }
    return result;
}

defineTypeNameAndDebug(MeshAdapter, 0);

std::vector<NeoN::localIdx> computeOffset(const Foam::fvMesh& mesh)
{
    std::vector<NeoN::localIdx> result;
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    result.push_back(0);
    // first all regular boundaries are collected
    forAll(bMesh, patchI)
    {
        NeoN::localIdx curOffset = result.back();
        const Foam::fvPatch& patch = bMesh[patchI];
        if (!Foam::isA<Foam::processorFvPatch>(patch))
        {
            result.push_back(curOffset + patch.size());
        }
    }
    forAll(bMesh, patchI)
    {
        NeoN::localIdx curOffset = result.back();
        const Foam::fvPatch& patch = bMesh[patchI];
        if (Foam::isA<Foam::processorFvPatch>(patch))
        {
            result.push_back(curOffset + patch.size());
        }
    }
    return result;
}

std::vector<NeoN::localIdx> computeNeighbRank(const Foam::fvMesh& mesh)
{
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    const Foam::lduInterfacePtrsList interfaces = bMesh.interfaces();

    auto result = std::vector<NeoN::localIdx>(); // bMesh.size(), -1);

    for (auto i = 0; i < interfaces.size(); i++)
    {
        if (interfaces.get(i) == nullptr)
        {
            continue;
        }
        if (Foam::isA<Foam::processorFvPatch>(interfaces[i]))
        {
            const Foam::processorFvPatch& patch =
                Foam::refCast<const Foam::processorFvPatch>(interfaces[i]);
            result.push_back(patch.neighbProcNo());
        }
    }
    return result;
}

std::vector<std::pair<NeoN::localIdx, NeoN::localIdx>>
computeNeighbRankAndSize(const Foam::fvMesh& mesh)
{
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    const Foam::lduInterfacePtrsList interfaces = bMesh.interfaces();
    auto result = std::vector<std::pair<NeoN::localIdx, NeoN::localIdx>>();

    for (auto i = 0; i < interfaces.size(); i++)
    {
        if (interfaces.get(i) == nullptr)
        {
            continue;
        }
        if (Foam::isA<Foam::processorFvPatch>(interfaces[i]))
        {
            const Foam::processorFvPatch& patch =
                Foam::refCast<const Foam::processorFvPatch>(interfaces[i]);
            result.emplace_back(patch.neighbProcNo(), patch.size());
        }
    }
    return result;
}

NeoN::localIdx computeProcBoundaryPatches(const Foam::fvMesh& mesh)
{
    NeoN::localIdx count = 0;
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    forAll(bMesh, patchI)
    {
        if (Foam::isA<Foam::processorFvPatch>(bMesh[patchI]))
        {
            count++;
        }
    }
    return count;
}

std::vector<NeoN::localIdx> computeNeighbourRank(const Foam::fvMesh& mesh)
{
    std::vector<NeoN::localIdx> result;
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    forAll(bMesh, patchI)
    {
        const auto* procPatch = Foam::isA<Foam::processorFvPatch>(bMesh[patchI]);
        if (procPatch)
        {
            result.push_back(static_cast<NeoN::localIdx>(procPatch->neighbProcNo()));
        }
    }
    return result;
}

int32_t computeNBoundaryFaces(const Foam::fvMesh& mesh)
{
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    int32_t nBoundaryFaces = 0;
    forAll(bMesh, patchI)
    {
        const Foam::fvPatch& patch = bMesh[patchI];
        nBoundaryFaces += patch.size();
    }
    return nBoundaryFaces;
}

NeoN::UnstructuredMesh
readOpenFOAMMesh(const NeoN::Executor exec, const Foam::fvMesh& mesh, bool fullMeshOnGPU)
{
    // Executor of "optional" fields
    const NeoN::Executor optExec = fullMeshOnGPU ? exec : NeoN::SerialExecutor {};

    Foam::scalarField magFaceAreas(mag(mesh.faceAreas()));

    Foam::labelList faceCells = flatBCField<Foam::labelList>(
        mesh,
        [](const Foam::fvPatch& patch) { return patch.faceCells(); }
    );
    Foam::vectorField cf =
        flatBCField<Foam::vectorField>(mesh, [](const Foam::fvPatch& patch) { return patch.Cf(); });
    Foam::vectorField cn = flatBCField<Foam::vectorField>(
        mesh,
        [](const Foam::fvPatch& patch) { return Foam::vectorField(patch.Cn()); }
    );
    Foam::vectorField sf =
        flatBCField<Foam::vectorField>(mesh, [](const Foam::fvPatch& patch) { return patch.Sf(); });
    Foam::scalarField magSf = flatBCField<Foam::scalarField>(
        mesh,
        [](const Foam::fvPatch& patch) { return patch.magSf(); }
    );
    Foam::vectorField nf = flatBCField<Foam::vectorField>(
        mesh,
        [](const Foam::fvPatch& patch) { return Foam::vectorField(patch.nf()); }
    );
    Foam::vectorField delta = flatBCField<Foam::vectorField>(
        mesh,
        [](const Foam::fvPatch& patch) { return Foam::vectorField(patch.delta()); }
    );
    Foam::scalarField weights = flatBCField<Foam::scalarField>(
        mesh,
        [](const Foam::fvPatch& patch) { return patch.weights(); }
    );
    Foam::scalarField deltaCoeffs = flatBCField<Foam::scalarField>(
        mesh,
        [](const Foam::fvPatch& patch) { return patch.deltaCoeffs(); }
    );
    std::vector<NeoN::localIdx> offset = computeOffset(mesh);

    // neighbourRank MUST be built in the same fvBoundaryMesh proc-patch order as computeOffset():
    // every consumer (BoundaryMesh::neighbourRankForRange, computeCommunicationPattern) indexes
    // neighbourRank_[k] by the k-th proc patch in offset order. computeNeighbRank() iterates
    // lduInterfacePtrsList (interface-index order, skipping nulls), which diverges from offset
    // order on non-X-split / sheared decompositions and yields the wrong peer rank. Use the
    // fvBoundaryMesh-ordered computeNeighbourRank() instead (D1 of the proc-halo comm-map fix).
    std::vector<NeoN::localIdx> neighbRank = computeNeighbourRank(mesh);
    NeoN::localIdx nProcPatches = neighbRank.size();
    NeoN::BoundaryMesh bMesh(
        exec,
        fromFoamField(exec, faceCells),
        fromFoamField(exec, cf),
        fromFoamField(exec, cn),
        fromFoamField(exec, sf),
        fromFoamField(exec, magSf),
        fromFoamField(exec, nf),
        fromFoamField(exec, delta),
        fromFoamField(exec, weights),
        fromFoamField(exec, deltaCoeffs),
        offset,
        nProcPatches,
        neighbRank
    );

    NeoN::UnstructuredMesh uMesh(
        exec,
        fromFoamField(optExec, mesh.points()),
        fromFoamField(exec, mesh.cellVolumes()),
        fromFoamField(exec, mesh.cellCentres()),
        fromFoamField(exec, mesh.faceAreas()),
        fromFoamField(exec, mesh.faceCentres()),
        fromFoamField(exec, magFaceAreas),
        fromFoamField(exec, mesh.faceOwner()),
        fromFoamField(exec, mesh.faceNeighbour()),
        bMesh
    );

    return uMesh;
}

MeshAdapter::MeshAdapter(const NeoN::Executor exec, const Foam::IOobject& io, const bool doInit)
    : fvMesh(io, doInit)
    , nfMesh_(readOpenFOAMMesh(exec, *this))
{
    if (doInit)
    {
        init(false); // do not initialise lower levels
    }
}


MeshAdapter::MeshAdapter(
    const NeoN::Executor exec,
    const Foam::IOobject& io,
    const Foam::zero,
    bool syncPar
)
    : fvMesh(io, Foam::zero {}, syncPar)
    , nfMesh_(readOpenFOAMMesh(exec, *this))
{}


MeshAdapter::MeshAdapter(
    const NeoN::Executor exec,
    const Foam::IOobject& io,
    Foam::pointField&& points,
    Foam::faceList&& faces,
    Foam::labelList&& allOwner,
    Foam::labelList&& allNeighbour,
    const bool syncPar
)
    : fvMesh(
        io,
        std::move(points),
        std::move(faces),
        std::move(allOwner),
        std::move(allNeighbour),
        syncPar
    )
    , nfMesh_(readOpenFOAMMesh(exec, *this))
{}


MeshAdapter::MeshAdapter(
    const NeoN::Executor exec,
    const Foam::IOobject& io,
    Foam::pointField&& points,
    Foam::faceList&& faces,
    Foam::cellList&& cells,
    const bool syncPar
)
    : fvMesh(io, std::move(points), std::move(faces), std::move(cells), syncPar)
    , nfMesh_(readOpenFOAMMesh(exec, *this))
{}

}
