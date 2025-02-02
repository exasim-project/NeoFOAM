// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoFOAM authors

#include "NeoFOAM/linearAlgebra/utilities.hpp"


namespace NeoFOAM::la
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec)
{
    return std::visit(
        [](auto concreteExec)
        { return gko::ext::kokkos::create_executor(concreteExec.underlyingExec()); },
        exec
    );
}

}
