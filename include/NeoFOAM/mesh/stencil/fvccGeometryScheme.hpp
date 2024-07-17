// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include "NeoFOAM/core/primitives/vector.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM
{
namespace fvcc = finiteVolume::cellCentred;

template<typename ValueType>
std::vector<fvcc::SurfaceBoundary<ValueType>> createCalculatedBCs(const UnstructuredMesh& mesh)
{
    const auto& bMesh = mesh.boundaryMesh();
    std::vector<fvcc::SurfaceBoundary<ValueType>> bcs;

    for (int patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        Dictionary patchDict({{"type", std::string("calculated")}});
        bcs.push_back(fvcc::SurfaceBoundary<ValueType>(mesh, patchDict, patchID));
    }

    return bcs;
};

class GeometrySchemeKernel
{
public:

    GeometrySchemeKernel(const UnstructuredMesh& mesh);

    virtual void updateWeights(const CPUExecutor& exec, fvcc::SurfaceField<scalar>& weights) = 0;
    virtual void updateWeights(const OMPExecutor& exec, fvcc::SurfaceField<scalar>& weights) = 0;
    virtual void updateWeights(const GPUExecutor& exec, fvcc::SurfaceField<scalar>& weights) = 0;

    virtual void
    updateDeltaCoeffs(const CPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs) = 0;
    virtual void
    updateDeltaCoeffs(const OMPExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs) = 0;
    virtual void
    updateDeltaCoeffs(const GPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs) = 0;

    virtual void updateNonOrthDeltaCoeffs(
        const CPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) = 0;
    virtual void updateNonOrthDeltaCoeffs(
        const OMPExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) = 0;
    virtual void updateNonOrthDeltaCoeffs(
        const GPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) = 0;

    virtual void updateNonOrthCorrectionVectors(
        const CPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    ) = 0;
    virtual void updateNonOrthCorrectionVectors(
        const OMPExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    ) = 0;
    virtual void updateNonOrthCorrectionVectors(
        const GPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    ) = 0;
};

class GeometryScheme
{
public:

    GeometryScheme(
        const Executor& exec,
        std::unique_ptr<GeometrySchemeKernel> kernel,
        const fvcc::SurfaceField<scalar>& weights,
        const fvcc::SurfaceField<scalar>& deltaCoeffs,
        const fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs,
        const fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    );

    GeometryScheme(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<GeometrySchemeKernel> kernel
    );

    GeometryScheme(const UnstructuredMesh& mesh // will lookup the kernel
    );

    const fvcc::SurfaceField<scalar>& weights() const;

    const fvcc::SurfaceField<scalar>& deltaCoeffs() const;

    const fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs() const;

    const fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors() const;

    void update();

    std::string name() const;

    // add selection mechanism via dictionary later
    static const std::shared_ptr<GeometryScheme> readOrCreate(const UnstructuredMesh& mesh);

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<GeometrySchemeKernel> kernel_;

    fvcc::SurfaceField<scalar> weights_;
    fvcc::SurfaceField<scalar> deltaCoeffs_;
    fvcc::SurfaceField<scalar> nonOrthDeltaCoeffs_;
    fvcc::SurfaceField<Vector> nonOrthCorrectionVectors_;
};


} // namespace NeoFOAM
