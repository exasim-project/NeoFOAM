// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include "NeoFOAM/core/primitives/vector.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/stencil/fvccGeometryScheme.hpp"

namespace NeoFOAM
{

class BasicFvccGeometryScheme : public FvccGeometrySchemeKernel
{
public:

    BasicFvccGeometryScheme(const UnstructuredMesh& uMesh);

    void updateWeights(const CPUExecutor& exec, fvcc::SurfaceField<scalar>& weights) override;
    void updateWeights(const OMPExecutor& exec, fvcc::SurfaceField<scalar>& weights) override;
    void updateWeights(const GPUExecutor& exec, fvcc::SurfaceField<scalar>& weights) override;

    void
    updateDeltaCoeffs(const CPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs) override;
    void
    updateDeltaCoeffs(const OMPExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs) override;
    void
    updateDeltaCoeffs(const GPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs) override;

    void updateNonOrthDeltaCoeffs(
        const CPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) override;
    void updateNonOrthDeltaCoeffs(
        const OMPExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) override;
    void updateNonOrthDeltaCoeffs(
        const GPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) override;

    void updateNonOrthCorrectionVectors(
        const CPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    ) override;
    void updateNonOrthCorrectionVectors(
        const OMPExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    ) override;
    void updateNonOrthCorrectionVectors(
        const GPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    ) override;

private:

    const UnstructuredMesh& uMesh_;
};

} // namespace NeoFOAM
