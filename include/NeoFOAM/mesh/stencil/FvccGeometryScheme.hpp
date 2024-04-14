// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include "NeoFOAM/primitives/vector.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/primitives/label.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccSurfaceField.hpp"

namespace NeoFOAM
{

class FvccGeometrySchemeKernel
{
public:

    FvccGeometrySchemeKernel(
        const unstructuredMesh& uMesh
    );

    virtual void updateWeights(const CPUExecutor& exec, fvccSurfaceField<scalar>& weights) = 0;
    virtual void updateWeights(const OMPExecutor& exec, fvccSurfaceField<scalar>& weights) = 0;
    virtual void updateWeights(const GPUExecutor& exec, fvccSurfaceField<scalar>& weights) = 0;

    virtual void updateDeltaCoeffs(const CPUExecutor& exec, fvccSurfaceField<scalar>& deltaCoeffs) = 0;
    virtual void updateDeltaCoeffs(const OMPExecutor& exec, fvccSurfaceField<scalar>& deltaCoeffs) = 0;
    virtual void updateDeltaCoeffs(const GPUExecutor& exec, fvccSurfaceField<scalar>& deltaCoeffs) = 0;

    virtual void updateNonOrthDeltaCoeffs(const CPUExecutor& exec, fvccSurfaceField<scalar>& nonOrthDeltaCoeffs) = 0;
    virtual void updateNonOrthDeltaCoeffs(const OMPExecutor& exec, fvccSurfaceField<scalar>& nonOrthDeltaCoeffs) = 0;
    virtual void updateNonOrthDeltaCoeffs(const GPUExecutor& exec, fvccSurfaceField<scalar>& nonOrthDeltaCoeffs) = 0;

    virtual void updateNonOrthCorrectionVectors(const CPUExecutor& exec, fvccSurfaceField<Vector>& nonOrthCorrectionVectors) = 0;
    virtual void updateNonOrthCorrectionVectors(const OMPExecutor& exec, fvccSurfaceField<Vector>& nonOrthCorrectionVectors) = 0;
    virtual void updateNonOrthCorrectionVectors(const GPUExecutor& exec, fvccSurfaceField<Vector>& nonOrthCorrectionVectors) = 0;

};

class FvccGeometryScheme
{
public:


    FvccGeometryScheme(
        const executor& exec,
        std::unique_ptr<FvccGeometrySchemeKernel> kernel,
        const fvccSurfaceField<scalar>& weights,
        const fvccSurfaceField<scalar>& deltaCoeffs,
        const fvccSurfaceField<scalar>& nonOrthDeltaCoeffs,
        const fvccSurfaceField<Vector>& nonOrthCorrectionVectors
    );

    FvccGeometryScheme(
        const executor& exec,
        const unstructuredMesh& uMesh,
        std::unique_ptr<FvccGeometrySchemeKernel> kernel
    );

    FvccGeometryScheme(
        const unstructuredMesh& uMesh // will lookup the kernel
    );

    const fvccSurfaceField<scalar>& weights() const;

    const fvccSurfaceField<scalar>& deltaCoeffs() const;

    const fvccSurfaceField<scalar>& nonOrthDeltaCoeffs() const;

    const fvccSurfaceField<Vector>& nonOrthCorrectionVectors() const;

    void update();

    std::string name() const;

    // TODO add selection mechanism via dictionary later
    static const std::shared_ptr<FvccGeometryScheme> readOrCreate(const unstructuredMesh& uMesh);

private:

    const executor exec_;
    const unstructuredMesh& uMesh_;
    std::unique_ptr<FvccGeometrySchemeKernel> kernel_;

    fvccSurfaceField<scalar> weights_;
    fvccSurfaceField<scalar> deltaCoeffs_;
    fvccSurfaceField<scalar> nonOrthDeltaCoeffs_;
    fvccSurfaceField<Vector> nonOrthCorrectionVectors_;
};



} // namespace NeoFOAM