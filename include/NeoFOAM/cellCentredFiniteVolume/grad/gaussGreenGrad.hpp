// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "Kokkos_Core.hpp"
#include <functional>


namespace NeoFOAM
{

struct GaussGreenKernel
{
    const unstructuredMesh& mesh_;
    const scalarField& phi_;
    vectorField& gradPhi_;

    GaussGreenKernel(const unstructuredMesh& mesh, const scalarField& phi, vectorField& gradPhi);

    void operator()(const GPUExecutor& exec);

    void operator()(const OMPExecutor& exec);

    void operator()(const CPUExecutor& exec);
};


class gaussGreenGrad
{
public:

    gaussGreenGrad(const executor& exec, const unstructuredMesh& mesh);

    const vectorField& grad(const scalarField& phi);

    void grad(vectorField& gradPhi, const scalarField& phi);

private:

    vectorField gradPhi_;
    const unstructuredMesh& mesh_;
};

} // namespace NeoFOAM