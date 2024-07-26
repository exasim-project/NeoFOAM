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
    std::cout << "Solving using Forward Euler" << std::endl;
    scalar dt = 0.001; // Time step
    fvcc::VolumeField<scalar>* refField = eqnSystem_.volumeField();
    // Field<scalar> Phi(eqnSystem_.exec(), eqnSystem_.nCells());
    // NeoFOAM::fill(Phi, 0.0);
    Field<scalar> source = eqnSystem_.explicitOperation();

    // for (auto& eqnTerm : eqnSystem_.temporalTerms())
    // {
    //     eqnTerm.temporalOperation(Phi);
    // }
    // Phi += source*dt;
    refField->internalField() -= source*dt;
    refField->correctBoundaryConditions();

    // check if execturo is GPU
    if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqnSystem_.exec()))
    {
       Kokkos::fence();
    }
}

std::unique_ptr<TimeIntegrationFactory> ForwardEuler::clone() const
{
    return std::make_unique<ForwardEuler>(*this);
}

} // namespace NeoFOAM
