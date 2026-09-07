// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "fvMesh.H"

#include <map>

#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/datastructures/databaseWrapper.hpp"
#include "NeoFOAM/compatibility/fvSchemes.hpp"

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
    /// Gradient operators built from gradSchemes, keyed by entry ("grad(p)"), see gradScheme()
    std::map<std::string, std::shared_ptr<fvcc::GradOperatorFactory<NeoN::Vec3>>> gradOps;
};


/**@brief Returns the gradSchemes-configured gradient operator for @p gradEntry.
 *
 * Built on first use and cached, so call sites need not hoist the operator out of the
 * time loop themselves. Lazy on purpose: solvers overwrite fvSchemesDict with
 * mapFvSchemes() after the RunTime is created.
 */
inline const std::shared_ptr<fvcc::GradOperatorFactory<NeoN::Vec3>>&
gradSchemePtr(RunTime& runTime, const std::string& gradEntry)
{
    auto& op = runTime.gradOps[gradEntry];
    if (!op)
    {
        op = makeGradOperator(runTime.exec, runTime.nfMesh, runTime.fvSchemesDict, gradEntry);
    }
    return op;
}

/**@brief As gradSchemePtr, for call sites that do not need to share ownership. */
inline const fvcc::GradOperatorFactory<NeoN::Vec3>&
gradScheme(RunTime& runTime, const std::string& gradEntry)
{
    return *gradSchemePtr(runTime, gradEntry);
}


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
