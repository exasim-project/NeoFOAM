// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

// This class shows how to implement a simple generator for Catch tests
class ExecutorGenerator final : public Catch::Generators::IGenerator<NeoFOAM::Executor>
{
public:

    int i = 1;
    std::vector<NeoFOAM::Executor> execs {NeoFOAM::CPUExecutor {}, NeoFOAM::GPUExecutor {}};
    NeoFOAM::Executor current_exec = NeoFOAM::SerialExecutor {};

    ExecutorGenerator() {}

    NeoFOAM::Executor const& get() const override;
    bool next() override
    {
        if (i >= 2) return false;
        current_exec = execs[i];
        i++;
        return true;
    }
};

// Avoids -Wweak-vtables
NeoFOAM::Executor const& ExecutorGenerator::get() const { return current_exec; }

Catch::Generators::GeneratorWrapper<NeoFOAM::Executor> allAvailableExecutor()
{
    return Catch::Generators::GeneratorWrapper<NeoFOAM::Executor>(
        // Another possibility:
        Catch::Detail::make_unique<ExecutorGenerator>()
    );
}
