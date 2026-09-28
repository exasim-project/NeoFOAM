// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include "NeoFOAM/auxiliary/setup.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/datastructures/databaseWrapper.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"

#include "fvc.H"

namespace NeoFOAM
{

void setDeltaT(Foam::Time& ofRunTime, RunTime& nfRunTime, Foam::scalar coNum)
{
    Foam::scalar maxDeltaTFact = nfRunTime.maxCo / (coNum + Foam::SMALL);
    Foam::scalar deltaTFact = Foam::min(Foam::min(maxDeltaTFact, 1.0 + 0.1 * maxDeltaTFact), 1.2);

    ofRunTime.setDeltaT(Foam::min(deltaTFact * ofRunTime.deltaTValue(), nfRunTime.maxDeltaT));
    NeoN::Logging::info("deltaT = {}", ofRunTime.deltaTValue());
}

void syncRunTimes(Foam::Time& ofRunTime, RunTime& nfRunTime, Foam::scalar coNum)
{
    if (nfRunTime.adjustTimeStep)
    {
        setDeltaT(ofRunTime, nfRunTime, coNum);
    }
    nfRunTime.dt = ofRunTime.deltaTValue();
    nfRunTime.t = ofRunTime.time().value();
}


std::unique_ptr<MeshAdapter> createMesh(const NeoN::Executor& exec, const Foam::Time& runTime)
{
    Foam::word regionName(Foam::polyMesh::defaultRegion);
    Foam::IOobject io(regionName, runTime.timeName(), runTime, Foam::IOobject::MUST_READ);
    return std::make_unique<MeshAdapter>(exec, io);
}

std::unique_ptr<Foam::fvMesh> createMesh(const Foam::Time& runTime)
{
    std::unique_ptr<Foam::fvMesh> meshPtr;
    Foam::word regionName(Foam::polyMesh::defaultRegion);
    NeoN::Logging::info("Create mesh for time = {}", runTime.timeOutputValue());

    meshPtr.reset(new Foam::fvMesh(
        Foam::IOobject(regionName, runTime.timeName(), runTime, Foam::IOobject::MUST_READ),
        false
    ));
    meshPtr->init(true); // initialise all (lower levels and current)

    return meshPtr;
}

/* @brief create a NeoN executor from a name
 * @return the Neon::Executor
 */
NeoN::Executor
createExecutor(const std::string execName, std::unique_ptr<NeoN::AllocatorStrategy> strategy)
{
    NeoN::Logging::info("Creating Executor {}", execName);
    if (execName == "Serial")
    {
        return NeoN::SerialExecutor(std::move(strategy));
    }
    if (execName == "CPU")
    {
        return NeoN::CPUExecutor(std::move(strategy));
    }
    if (execName == "GPU")
    {
        return NeoN::GPUExecutor(std::move(strategy));
    }
    if (execName == "default")
    {
        return NeoN::createDefaultExecutor(std::move(strategy));
    }


    Foam::FatalError << "unknown Executor: " << execName << Foam::nl
                     << "Available executors: Serial, CPU, GPU, default" << Foam::nl
                     << Foam::abort(Foam::FatalError);

    return NeoN::SerialExecutor();
}

NeoN::Executor createExecutor(const Foam::word& execName)
{
    // Declared in setup.hpp; create the named executor with the default allocator.
    // Lets callers select an executor by name without a controlDict round-trip
    // (used by the create_adapter_run_time Python binding to default to Serial).
    return createExecutor(std::string(execName), std::make_unique<NeoN::DefaultAllocator>());
}

NeoN::Executor createExecutor(const Foam::dictionary& dict)
{
    auto execName = std::string(dict.get<Foam::word>("executor"));
    auto allocator = std::string(dict.get<Foam::word>("allocator"));
    if (allocator == "Umpire")
    {
        return createExecutor(execName, std::make_unique<NeoN::UmpireAllocator>());
    }
    if (allocator == "UmpirePool")
    {
        // TODO allow percentual pool size
        auto poolSizeGB = Foam::scalar(dict.get<Foam::scalar>("memPoolSize"));
        NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::MemorySpace::GPU, poolSizeGB * 1e9);
        return createExecutor(execName, std::make_unique<NeoN::UmpirePoolAllocator>());
    }
    return createExecutor(execName, std::make_unique<NeoN::DefaultAllocator>());
}

NeoN::Executor createExecutor(const Foam::argList& args, const Foam::dictionary& dict)
{
    std::string execName = args.found("executor")
                             ? std::string(args.get<Foam::word>("executor"))
                             : std::string(dict.getOrDefault<Foam::word>("executor", "Serial"));

    if (dict.found("allocator"))
    {
        auto allocator = std::string(dict.get<Foam::word>("allocator"));
        if (allocator == "Umpire")
            return createExecutor(execName, std::make_unique<NeoN::UmpireAllocator>());
        if (allocator == "UmpirePool")
        {
            auto poolSizeGB = Foam::scalar(dict.get<Foam::scalar>("memPoolSize"));
            NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::MemorySpace::GPU, poolSizeGB * 1e9);
            return createExecutor(execName, std::make_unique<NeoN::UmpirePoolAllocator>());
        }
    }
    return createExecutor(execName, std::make_unique<NeoN::DefaultAllocator>());
}

NeoFOAM::RunTime createAdapterRunTime(const Foam::Time& in)
{
    auto exec = createExecutor(in.controlDict());
    return createAdapterRunTime(in, exec);
}

NeoFOAM::RunTime createAdapterRunTime(const Foam::Time& in, const Foam::argList& args)
{
    auto exec = createExecutor(args, in.controlDict());
    return createAdapterRunTime(in, exec);
}

RunTime createAdapterRunTime(const Foam::Time& in, const NeoN::Executor exec)
{
    NeoN::Logging::info("Creating NeoFOAM runTime");

    // If a plain fvMesh is already registered under the default region name (e.g. from
    // test harness), check it out so the MeshAdapter can register itself in its place.
    // The original object remains alive via its owning unique_ptr; only the registry
    // entry is removed.
    if (in.foundObject<Foam::fvMesh>(Foam::polyMesh::defaultRegion))
    {
        Foam::fvMesh& existing =
            const_cast<Foam::fvMesh&>(in.lookupObject<Foam::fvMesh>(Foam::polyMesh::defaultRegion));
        if (!dynamic_cast<MeshAdapter*>(&existing))
        {
            in.objectRegistry::checkOut(Foam::polyMesh::defaultRegion);
        }
    }

    std::unique_ptr<MeshAdapter> meshPtr = createMesh(exec, in);
    MeshAdapter& mesh = *meshPtr;
    auto mpiEnvironment = NeoN::mpi::Environment {};

    auto& nfMesh = mesh.nfMesh();
    RunTime rt {
        .db = NeoN::Database(),
        .meshPtr = std::move(meshPtr),
        .mesh = mesh,
        .nfMesh = mesh.nfMesh(),
        .exec = exec,
        .t = in.time().value(),
        .dt = in.deltaT().value(),
        .adjustTimeStep = in.controlDict().getOrDefault("adjustTimeStep", false),
        .maxCo = in.controlDict().getOrDefault<Foam::scalar>("maxCo", 1),
        .maxDeltaT = in.controlDict().getOrDefault<Foam::scalar>("maxDeltaT", Foam::GREAT),
        .controlDict = convert(in.controlDict()),
        .fvSolutionDict = convert(mesh.solutionDict()),
        .fvSchemesDict = convert(mesh.schemesDict()),
        .mpiEnvironment = mpiEnvironment
    };
    rt.dbWrapper = std::make_unique<DatabaseWrapper>(in, rt.db);
    return rt;
}

}
