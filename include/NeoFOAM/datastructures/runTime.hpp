// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "fvMesh.H"

#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"

namespace NeoFOAM
{

/* @brief A struct holding typical runTime information*/
struct RunTime
{
    NeoN::Database db;
    std::unique_ptr<MeshAdapter> meshPtr;
    MeshAdapter& mesh;
    NeoN::UnstructuredMesh& nfMesh;
    NeoN::Executor exec;
    Foam::scalar t;
    Foam::scalar dt;
    bool adjustTimeStep;
    Foam::scalar maxCo;
    Foam::scalar maxDeltaT;
    NeoN::Dictionary controlDict;
    NeoN::Dictionary fvSolutionDict;
    NeoN::Dictionary fvSchemesDict;
};
} // End namespace NeoFOAM
