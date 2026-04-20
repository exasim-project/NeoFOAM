// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"


#include "processorFvPatch.H"
#include "lduInterfaceField.H"
#include "Pstream.H"
#include <mpi.h>

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
    forAll(bMesh, patchI)
    {
        NeoN::localIdx curOffset = result.back();
        const Foam::fvPatch& patch = bMesh[patchI];
        result.push_back(curOffset + patch.size());
    }
    return result;
}

std::vector<NeoN::localIdx> computeNeighbRank(const Foam::fvMesh& mesh)
{
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    const Foam::lduInterfacePtrsList interfaces = bMesh.interfaces();


    auto result = std::vector<NeoN::localIdx>(bMesh.size(), -1);

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
            result[i] = patch.neighbProcNo();
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

// FIXME
NeoN::Vector<NeoN::localIdx> computeIsProc(const NeoN::Executor& exec, const Foam::fvMesh& mesh)
{
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    const Foam::lduInterfacePtrsList interfaces = bMesh.interfaces();
    NeoN::localIdx nFaces = computeNBoundaryFaces(mesh);

    auto result = std::vector<NeoN::localIdx>(nFaces, 0);

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
            // FIXME -1 for non-owner/owner
            result[i] = -1;
        }
    }

    return NeoN::Vector<NeoN::localIdx>(exec, result);
}

// Compute the CommunicationPattern from OpenFOAM mesh data.
// Uses MPI_Sendrecv to exchange local face-cell indices with each processor neighbor,
// then computes global indices from the rank offsets gathered via MPI_Allgather.
NeoN::CommunicationPattern computeOpenFOAMCommunicationPattern(const Foam::fvMesh& mesh)
{
    if (!Foam::Pstream::parRun())
        return {};

    NeoN::mpi::Environment mpiEnviron;
    const int nRanks = static_cast<int>(mpiEnviron.sizeRank());
    const int myRank = static_cast<int>(mpiEnviron.rank());

    // 1. Gather cell counts from all ranks to compute global offsets.
    const int localNCells = mesh.nCells();
    std::vector<int> allNCells(nRanks);
    MPI_Allgather(&localNCells, 1, MPI_INT, allNCells.data(), 1, MPI_INT, mpiEnviron.comm());

    std::vector<int> globalOffset(nRanks + 1, 0);
    for (int i = 0; i < nRanks; i++)
        globalOffset[i + 1] = globalOffset[i] + allNCells[i];

    // 2. For each processor boundary patch, exchange face-cell global indices with the neighbor.
    const Foam::fvBoundaryMesh& bMesh = mesh.boundary();
    const Foam::lduInterfacePtrsList interfaces = bMesh.interfaces();

    auto sendCounts = std::vector<int>(nRanks + 1, 0);
    auto recvIdx = std::vector<int>();

    for (auto i = 0; i < interfaces.size(); i++)
    {
        if (interfaces.get(i) == nullptr)
            continue;
        if (!Foam::isA<Foam::processorFvPatch>(interfaces[i]))
            continue;

        const Foam::processorFvPatch& patch =
            Foam::refCast<const Foam::processorFvPatch>(interfaces[i]);
        const int neighbRank = patch.neighbProcNo();
        const int nFaces = patch.size();

        sendCounts[neighbRank] += nFaces;
        sendCounts[nRanks] += nFaces;

        // Build global indices of my face cells, then exchange with neighbor.
        const Foam::labelList& myFaceCells = patch.faceCells();
        std::vector<int> myGlobalCells(nFaces);
        for (int j = 0; j < nFaces; j++)
            myGlobalCells[j] = static_cast<int>(myFaceCells[j]) + globalOffset[myRank];

        std::vector<int> neighGlobalCells(nFaces);
        MPI_Sendrecv(
            myGlobalCells.data(),
            nFaces,
            MPI_INT,
            neighbRank,
            0,
            neighGlobalCells.data(),
            nFaces,
            MPI_INT,
            neighbRank,
            0,
            mpiEnviron.comm(),
            MPI_STATUS_IGNORE
        );

        recvIdx.insert(recvIdx.end(), neighGlobalCells.begin(), neighGlobalCells.end());
    }

    std::vector<NeoN::localIdx> boundaryMapVector;
    return {sendCounts, recvIdx, boundaryMapVector, mpiEnviron};
}

NeoN::CommunicationPattern createCommunicationPattern(const RunTime& runTime)
{
    if (runTime.nfMesh.stencilDB().contains("communicationPattern"))
        return runTime.nfMesh.stencilDB().get<NeoN::CommunicationPattern>("communicationPattern");
    return NeoN::computeCommunicationPattern(runTime.nfMesh);
}

NeoN::UnstructuredMesh
readOpenFOAMMesh(const NeoN::Executor exec, const Foam::fvMesh& mesh, bool fullMeshOnGPU)
{
    const int32_t nCells = mesh.nCells();
    const int32_t nInternalFaces = mesh.nInternalFaces();
    const int32_t nBoundaryFaces = computeNBoundaryFaces(mesh);
    const int32_t nBoundaries = mesh.boundary().size();
    const int32_t nFaces = mesh.nFaces();

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

    std::vector<NeoN::localIdx> neighbourRank = computeNeighbRank(mesh);
    auto procPatches = computeNeighbRankAndSize(mesh);
    NeoN::localIdx nProcFaces = 0;
    for (const auto& [rank, size] : procPatches)
        nProcFaces += size;
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
        nProcFaces,
        neighbourRank
    );

    NeoN::UnstructuredMesh uMesh(
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

    if (Foam::Pstream::parRun())
    {
        uMesh.stencilDB().insert(
            "communicationPattern", computeOpenFOAMCommunicationPattern(mesh)
        );
    }

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
