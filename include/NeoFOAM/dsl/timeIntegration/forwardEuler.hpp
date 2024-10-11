// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/timeIntegration/timeIntegration.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

class ForwardEuler : public TimeIntegrationFactory::Register<ForwardEuler>
{

public:

    ForwardEuler(const dsl::Equation& eqnSystem, const Dictionary& dict);

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "forwardEuler timeIntegration"; }

    static std::string schema() { return "none"; }

    void solve() override;

    std::unique_ptr<TimeIntegrationFactory> clone() const override;
};

} // namespace NeoFOAM
