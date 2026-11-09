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


// METHOD that convert or build openfom mesh data to array of NeoN::Vector and return the
// TrackingMesh data structure return the 15 NeoN::Vector of the TrackingMesh data structure
// everything need for the locate walk method
TrackingMesh buildTrackingMesh(const NeoN::Executor& exec, const Foam::fvMesh& mesh)
{
    using NeoN::label;

    // Get the number of cells, faces and internal faces from the OpenFOAM mesh
    const label nCells = mesh.nCells();
    const label nFaces = mesh.nFaces();
    const label nInternal = mesh.nInternalFaces();

    // faces of every cell (internal and boundary), compressed row storage
    std::vector<label> offsets(nCells + 1, 0);
    std::vector<label> faces;
    const Foam::cellList& cells = mesh.cells();
    // loop over all cells and store the faces of each cell in the faces vector and the offsets of
    // each cell in the offsets vector
    for (label c = 0; c < nCells; c++)
    {
        for (const Foam::label f : cells[c])
        {
            faces.push_back(f);
        }
        offsets[c + 1] = static_cast<label>(faces.size());
    }

    std::vector<label> owner(nFaces), neighbour(nFaces, -1), type(nFaces, escapeFace);
    std::vector<label> partner(nFaces, -1);
    std::vector<NeoN::Vec3> centres(nFaces), areas(nFaces), shift(nFaces, NeoN::Vec3(0, 0, 0));
    for (label f = 0; f < nFaces; f++)
    {
        owner[f] = mesh.faceOwner()[f];
        centres[f] = toVec3(mesh.faceCentres()[f]);
        areas[f] = toVec3(mesh.faceAreas()[f]);
    }
    for (label f = 0; f < nInternal; f++)
    {
        neighbour[f] = mesh.faceNeighbour()[f];
        type[f] = internalFace;
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
            type[f] = cyclicFace;
            partner[f] = mesh.faceOwner()[fn];
            shift[f] = centres[fn] - centres[f];
        }
    }

    // points of every face (compressed rows) and OpenFOAM's tet base point of each face
    std::vector<label> fpOffsets(nFaces + 1, 0), fpPoints, basePt(nFaces, 0);
    const Foam::labelList& tetBase = mesh.tetBasePtIs();
    for (label f = 0; f < nFaces; f++)
    {
        for (const Foam::label p : mesh.faces()[f])
        {
            fpPoints.push_back(p);
        }
        fpOffsets[f + 1] = static_cast<label>(fpPoints.size());
        basePt[f] = (tetBase[f] < 0) ? 0 : tetBase[f]; // as tetIndices::faceTriIs
    }

    std::vector<NeoN::Vec3> cellCentres(nCells), points(mesh.nPoints());
    std::vector<NeoN::scalar> volumes(nCells);
    for (label c = 0; c < nCells; c++)
    {
        cellCentres[c] = toVec3(mesh.cellCentres()[c]);
        volumes[c] = mesh.cellVolumes()[c];
    }
    for (label p = 0; p < mesh.nPoints(); p++)
    {
        points[p] = toVec3(mesh.points()[p]);
    }

    return TrackingMesh {
        NeoN::Vector<label>(exec, offsets),
        NeoN::Vector<label>(exec, faces),
        NeoN::Vector<label>(exec, fpOffsets),
        NeoN::Vector<label>(exec, fpPoints),
        NeoN::Vector<label>(exec, basePt),
        NeoN::Vector<label>(exec, owner),
        NeoN::Vector<label>(exec, neighbour),
        NeoN::Vector<NeoN::Vec3>(exec, centres),
        NeoN::Vector<NeoN::Vec3>(exec, areas),
        NeoN::Vector<label>(exec, type),
        NeoN::Vector<label>(exec, partner),
        NeoN::Vector<NeoN::Vec3>(exec, shift),
        NeoN::Vector<NeoN::Vec3>(exec, cellCentres),
        NeoN::Vector<NeoN::scalar>(exec, volumes),
        NeoN::Vector<NeoN::Vec3>(exec, points)
    };
}

} // namespace NeoFOAM::lagrangian
