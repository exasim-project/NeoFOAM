// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "fvMesh.H"

#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/datastructures/databaseWrapper.hpp"

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
    NeoN::mpi::Environment mpiEnvironment;
    std::unique_ptr<DatabaseWrapper> dbWrapper; ///< Registers db in Foam::Time objectRegistry
};


/**@brief convenience function to avoid recreating objects by storing them in the runtime db*/
template<typename RegisteredType, typename InitializerType>
RegisteredType& readOrCreate(RunTime& runTime, std::string name, InitializerType init)
{
    if (!runTime.controlDict.contains(name))
    {
        runTime.controlDict.insert(std::string(name), init());
    }
    return runTime.controlDict.get<RegisteredType>(name);
}

} // End namespace NeoFOAM
