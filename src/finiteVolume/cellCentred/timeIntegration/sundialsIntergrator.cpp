// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/sundialsIntergrator.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


SundailsIntergrator::SundailsIntergrator(const dsl::EqnSystem& eqnSystem, const Dictionary& dict)
    : TimeIntegrationFactory::Register<SundailsIntergrator>(eqnSystem, dict)
{
    // Constructor
}

void SundailsIntergrator::solve()
{
    // Solve function
}

std::unique_ptr<TimeIntegrationFactory> SundailsIntergrator::clone() const
{
    return std::make_unique<SundailsIntergrator>(*this);
}

} // namespace NeoFOAM
