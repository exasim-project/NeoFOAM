// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/forwardEuler.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


ForwardEuler::ForwardEuler(const dsl::EqnSystem& eqnSystem, const Dictionary& dict)
    : TimeIntegrationFactory::Register<ForwardEuler>(eqnSystem, dict)
{
    // Constructor
}

void ForwardEuler::solve()
{
    // Solve function
}

std::unique_ptr<TimeIntegrationFactory> ForwardEuler::clone() const
{
    return std::make_unique<ForwardEuler>(*this);
}

} // namespace NeoFOAM
