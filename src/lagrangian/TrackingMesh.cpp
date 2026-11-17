// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/lagrangian/TrackingMesh.hpp"
#include "cyclicPolyPatch.H"
#include <vector>

namespace NeoFOAM::lagrangian
{

namespace
// Convert OpenFOAM's vector to NeoN::Vec3 format
{
NeoN::Vec3 toVec3(const Foam::vector& v) { return NeoN::Vec3(v.x(), v.y(), v.z()); }
}


// Read from the OpenFOAM mesh what NeoN's UnstructuredMesh lacks for the locate walk and
// the cellPoint interpolation, and build NeoN's TrackingMesh from it.
NeoN::TrackingMesh buildTrackingMesh(const NeoN::Executor& exec, const MeshAdapter& mesh)
{
    using NeoN::label;
    using NeoN::localIdx;

    if (!(exec == mesh.nfMesh().exec()))
    {
        NF_ERROR_EXIT("buildTrackingMesh: the executor must be the executor of the NeoN mesh");
    }

    // Get the number of cells, faces and internal faces from the OpenFOAM mesh
    const label nCells = mesh.nCells();
    const label nFaces = mesh.nFaces();

    // faces of every cell (internal and boundary), as a segmented list
    std::vector<localIdx> cellFaceSegments(nCells + 1, 0);
    std::vector<label> cellFaces;
    const Foam::cellList& cells = mesh.cells();
    for (label c = 0; c < nCells; c++)
    {
        for (const Foam::label f : cells[c])
        {
            cellFaces.push_back(f);
        }
        cellFaceSegments[c + 1] = static_cast<localIdx>(cellFaces.size());
    }

    // face type: internal for internal faces, escape for boundary faces unless cyclic
    std::vector<label> type(nFaces, label(NeoN::TrackingFaceType::escape));
    std::vector<label> partner(nFaces, -1);
    std::vector<NeoN::Vec3> shift(nFaces, NeoN::Vec3(0, 0, 0));
    for (label f = 0; f < mesh.nInternalFaces(); f++)
    {
        type[f] = label(NeoN::TrackingFaceType::internal);
    }

    // periodic patches: pair face i of a cyclic patch with face i of its neighbour patch
    for (const Foam::polyPatch& pp : mesh.boundaryMesh())
    {
        const auto* cyc = Foam::isA<Foam::cyclicPolyPatch>(pp);
        if (!cyc)
        {
            continue;
        }
        if (!cyc->parallel())
        {
            FatalErrorInFunction << "cyclic patch " << pp.name()
                                 << ": only translational cyclics are supported by the "
                                    "particle tracking"
                                 << Foam::exit(Foam::FatalError);
        }
        const Foam::cyclicPolyPatch& nbr = cyc->neighbPatch();
        for (label i = 0; i < cyc->size(); i++)
        {
            const label f = cyc->start() + i;
            const label fn = nbr.start() + i;
            type[f] = label(NeoN::TrackingFaceType::cyclic);
            partner[f] = mesh.faceOwner()[fn];
            shift[f] = toVec3(mesh.faceCentres()[fn]) - toVec3(mesh.faceCentres()[f]);
        }
    }

    // points of every face (segmented list) and OpenFOAM's tet base point of each face
    std::vector<localIdx> facePointSegments(nFaces + 1, 0);
    std::vector<label> facePoints, basePt(nFaces, 0);
    const Foam::labelList& tetBase = mesh.tetBasePtIs();
    for (label f = 0; f < nFaces; f++)
    {
        for (const Foam::label p : mesh.faces()[f])
        {
            facePoints.push_back(p);
        }
        facePointSegments[f + 1] = static_cast<localIdx>(facePoints.size());
        basePt[f] = (tetBase[f] < 0) ? 0 : tetBase[f]; // as tetIndices::faceTriIs
    }

    // points on the executor (UnstructuredMesh may keep them on the host only)
    std::vector<NeoN::Vec3> points(mesh.nPoints());
    for (label p = 0; p < mesh.nPoints(); p++)
    {
        points[p] = toVec3(mesh.points()[p]);
    }

    return NeoN::TrackingMesh(
        mesh.nfMesh(),
        NeoN::SegmentedVector<label, localIdx>(
            NeoN::Vector<label>(exec, cellFaces),
            NeoN::Vector<localIdx>(exec, cellFaceSegments)
        ),
        NeoN::SegmentedVector<label, localIdx>(
            NeoN::Vector<label>(exec, facePoints),
            NeoN::Vector<localIdx>(exec, facePointSegments)
        ),
        NeoN::Vector<label>(exec, basePt),
        NeoN::Vector<label>(exec, type),
        NeoN::Vector<label>(exec, partner),
        NeoN::Vector<NeoN::Vec3>(exec, shift),
        NeoN::Vector<NeoN::Vec3>(exec, points)
    );
}

} // namespace NeoFOAM::lagrangian
