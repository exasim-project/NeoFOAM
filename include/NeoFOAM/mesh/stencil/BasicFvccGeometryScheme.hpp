// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include "NeoFOAM/primitives/vector.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/primitives/label.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccSurfaceField.hpp"
#include "NeoFOAM/mesh/stencil/FvccGeometryScheme.hpp"

namespace NeoFOAM
{

class BasicFvccGeometryScheme : public FvccGeometrySchemeKernel
{
public:

    BasicFvccGeometryScheme(const unstructuredMesh& uMesh);

    void updateWeights(const CPUExecutor& exec, fvccSurfaceField<scalar>& weights) override;
    void updateWeights(const OMPExecutor& exec, fvccSurfaceField<scalar>& weights) override;
    void updateWeights(const GPUExecutor& exec, fvccSurfaceField<scalar>& weights) override;

    void updateDeltaCoeffs(const CPUExecutor& exec, fvccSurfaceField<scalar>& deltaCoeffs) override;
    void updateDeltaCoeffs(const OMPExecutor& exec, fvccSurfaceField<scalar>& deltaCoeffs) override;
    void updateDeltaCoeffs(const GPUExecutor& exec, fvccSurfaceField<scalar>& deltaCoeffs) override;

    void updateNonOrthDeltaCoeffs(const CPUExecutor& exec, fvccSurfaceField<scalar>& nonOrthDeltaCoeffs) override;
    void updateNonOrthDeltaCoeffs(const OMPExecutor& exec, fvccSurfaceField<scalar>& nonOrthDeltaCoeffs) override;
    void updateNonOrthDeltaCoeffs(const GPUExecutor& exec, fvccSurfaceField<scalar>& nonOrthDeltaCoeffs) override;

    void updateNonOrthCorrectionVectors(const CPUExecutor& exec, fvccSurfaceField<Vector>& nonOrthCorrectionVectors) override;
    void updateNonOrthCorrectionVectors(const OMPExecutor& exec, fvccSurfaceField<Vector>& nonOrthCorrectionVectors) override;
    void updateNonOrthCorrectionVectors(const GPUExecutor& exec, fvccSurfaceField<Vector>& nonOrthCorrectionVectors) override;

private:

    const unstructuredMesh& uMesh_;
};

} // namespace NeoFOAM
