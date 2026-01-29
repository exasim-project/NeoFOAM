// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>
#include <memory>

#include "NeoN/NeoN.hpp"

#include "fvMesh.H"

namespace NeoFOAM
{

std::vector<NeoN::localIdx> computeOffset(const Foam::fvMesh& mesh);

int32_t computeNBoundaryFaces(const Foam::fvMesh& mesh);

template<typename FieldT>
FieldT flatBCField(const Foam::fvMesh& mesh, std::function<FieldT(const Foam::fvPatch&)> f);

/** @brief read OpenFOAM mesh and construct NeoN mesh
    * @param exec the executor where to store mesh data
    * @param mesh the OpenFOAM mesh
    * @param fullMeshOnGPU - if false only owner,neighbour,faceAreas and cellVolumes are placed on GPU
    */
NeoN::UnstructuredMesh readOpenFOAMMesh(const NeoN::Executor exec, const Foam::fvMesh& mesh, bool fullMeshOnGPU=false);

/** @class MeshAdapter
 */
class MeshAdapter : public Foam::fvMesh
{
    using word = Foam::word;

    NeoN::UnstructuredMesh nfMesh_;

    // Private Member Functions

    //- No copy construct
    MeshAdapter(const MeshAdapter&) = delete;

    //- No copy assignment
    void operator=(const MeshAdapter&) = delete;

public:

    //- Runtime type information
    TypeName("MeshAdapter");

    // Constructors

    //- Construct from IOobject
    explicit MeshAdapter(
        const NeoN::Executor exec,
        const Foam::IOobject& io,
        const bool doInit = true
    );

    //- Construct from IOobject or as zero-sized mesh
    //  Boundary is added using addFvPatches() member function
    MeshAdapter(
        const NeoN::Executor exec,
        const Foam::IOobject& io,
        const Foam::zero,
        bool syncPar = true
    );

    //- Construct from components without boundary.
    //  Boundary is added using addFvPatches() member function
    MeshAdapter(
        const NeoN::Executor exec,
        const Foam::IOobject& io,
        Foam::pointField&& points,
        Foam::faceList&& faces,
        Foam::labelList&& allOwner,
        Foam::labelList&& allNeighbour,
        const bool syncPar = true
    );

    //- Construct without boundary from cells rather than owner/neighbour.
    //  Boundary is added using addPatches() member function
    MeshAdapter(
        const NeoN::Executor exec,
        const Foam::IOobject& io,
        Foam::pointField&& points,
        Foam::faceList&& faces,
        Foam::cellList&& cells,
        const bool syncPar = true
    );

    //- Destructor
    virtual ~MeshAdapter() = default;

    NeoN::UnstructuredMesh& nfMesh() { return nfMesh_; }

    const NeoN::UnstructuredMesh& nfMesh() const { return nfMesh_; }

    const NeoN::Executor exec() const { return nfMesh().exec(); }
};

std::unique_ptr<MeshAdapter> createMesh(const NeoN::Executor& exec, const Foam::Time& runTime);

std::unique_ptr<Foam::fvMesh> createMesh(const Foam::Time& runTime);

} // End namespace Foam
