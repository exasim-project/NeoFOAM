// SPDX-FileCopyrightText: 2024-2026 NeoFOAM authors
// SPDX-License-Identifier: Unlicense

#pragma once

#include <memory>
#include <string>
#include <vector>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/compatibility/fvSchemes.hpp"
#include "NeoFOAM/auxiliary/setup.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

// OpenFOAM headers
#include "fvCFD.H"
#include "pisoControl.H"

namespace nf = NeoFOAM;

// ===========================================================================
// Building block 1: Runtime — OpenFOAM Time + NeoFOAM RunTime adapter
// ===========================================================================
class Runtime
{
    std::vector<std::string> argStrings_;
    std::vector<char*> cArgs_;
    char** argv_ = nullptr; // lvalue needed by Foam::argList(int&, char**&)
    int argc_ = 0;

    std::unique_ptr<Foam::argList> args_;
    Foam::Time* foamTime_ = nullptr;
    std::unique_ptr<nf::RunTime> rt_;

public:

    explicit Runtime(std::vector<std::string> argv)
    {
        argStrings_ = std::move(argv);
        argc_ = static_cast<int>(argStrings_.size());
        cArgs_.resize(argc_ + 1, nullptr);
        for (int i = 0; i < argc_; ++i)
            cArgs_[i] = const_cast<char*>(argStrings_[i].c_str());
        cArgs_[argc_] = nullptr;
        argv_ = cArgs_.data();

        args_ = std::make_unique<Foam::argList>(argc_, argv_);
        foamTime_ = new Foam::Time(Foam::Time::controlDictName, *args_);
        rt_ = std::make_unique<nf::RunTime>(nf::createAdapterRunTime(*foamTime_));

        // Map fvSolution solver entries for Ginkgo
        auto& solverDict = rt_->fvSolutionDict.subDict("solvers");
        for (auto& name : solverDict.keys())
        {
            solverDict.subDict(name) = nf::mapFvSolution(solverDict.subDict(name));
        }

        // Map fvSchemes (mirrors C++ neoIcoFoam.cpp line: schemesDict = nf::mapFvSchemes(...))
        rt_->fvSchemesDict = nf::mapFvSchemes(rt_->fvSchemesDict);
    }

    ~Runtime()
    {
        rt_.reset();
        delete foamTime_;
        foamTime_ = nullptr;
        args_.reset();
    }

    // Time loop
    bool loop() { return foamTime_->loop(); }
    double time() const { return foamTime_->time().value(); }
    double deltaT() const { return foamTime_->deltaTValue(); }
    std::string timeName() const { return std::string(foamTime_->timeName()); }

    // Sync NeoN runtime from OF (time, deltaT, adjustable timestep)
    void sync(double coNum) { nf::syncRunTimes(*foamTime_, *rt_, coNum); }

    // IO
    void write() { foamTime_->write(); }
    bool outputTime() { return foamTime_->outputTime(); }
    void printExecutionTime() { foamTime_->printExecutionTime(Foam::Info); }

    // Access NeoFOAM RunTime (for PDESolver, field registration, ...)
    nf::RunTime& nfRuntime() { return *rt_; }
    const nf::RunTime& nfRuntime() const { return *rt_; }

    // Convenience accessors forwarding into nf::RunTime
    const NeoN::Executor& executor() const { return rt_->exec; }
    NeoN::UnstructuredMesh& nfMesh() { return rt_->nfMesh; }
    const NeoN::UnstructuredMesh& nfMesh() const { return rt_->nfMesh; }
    NeoN::Database& db() { return rt_->db; }
    nf::MeshAdapter& mesh() { return rt_->mesh; }
    const nf::MeshAdapter& mesh() const { return rt_->mesh; }

    // Low-level OF Time (needed by PisoControl, field readers)
    Foam::Time& foamTime() { return *foamTime_; }
};

// ===========================================================================
// Building block 2: PisoControl — wraps Foam::pisoControl
// ===========================================================================
class PisoControl
{
    std::unique_ptr<Foam::pisoControl> piso_;

public:

    explicit PisoControl(Runtime& rt)
        : piso_(std::make_unique<Foam::pisoControl>(rt.mesh()))
    {}

    bool momentumPredictor() { return piso_->momentumPredictor(); }
    bool correct() { return piso_->correct(); }
    bool correctNonOrthogonal() { return piso_->correctNonOrthogonal(); }
    bool finalNonOrthogonalIter() { return piso_->finalNonOrthogonalIter(); }
};
