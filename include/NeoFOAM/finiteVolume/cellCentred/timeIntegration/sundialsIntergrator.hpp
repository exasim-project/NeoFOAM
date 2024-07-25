// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{


class SundailsIntergrator : public TimeIntegrationFactory::Register<SundailsIntergrator>
{

public:

    SundailsIntergrator(const dsl::EqnSystem& eqnSystem, const Dictionary& dict);

    static std::string name() { return "sundialsIntergrator"; }

    static std::string doc() { return "sundialsIntergrator timeIntegration"; }

    static std::string schema() { return "none"; }

    void solve() override;

    std::unique_ptr<TimeIntegrationFactory> clone() const override;
};

} // namespace NeoFOAM
