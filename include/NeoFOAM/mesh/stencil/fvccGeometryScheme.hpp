// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include "NeoFOAM/core/primitives/vector.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/primitives/label.hpp"
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

class FvccGeometrySchemeKernel
{
public:

    FvccGeometrySchemeKernel(const UnstructuredMesh& uMesh);

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

class FvccGeometryScheme
{
public:


    FvccGeometryScheme(
        const Executor& exec,
        std::unique_ptr<FvccGeometrySchemeKernel> kernel,
        const fvcc::SurfaceField<scalar>& weights,
        const fvcc::SurfaceField<scalar>& deltaCoeffs,
        const fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs,
        const fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
    );

    FvccGeometryScheme(
        const Executor& exec,
        const UnstructuredMesh& uMesh,
        std::unique_ptr<FvccGeometrySchemeKernel> kernel
    );

    FvccGeometryScheme(const UnstructuredMesh& uMesh // will lookup the kernel
    );

    const fvcc::SurfaceField<scalar>& weights() const;

    const fvcc::SurfaceField<scalar>& deltaCoeffs() const;

    const fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs() const;

    const fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors() const;

    void update();

    std::string name() const;

    // add selection mechanism via dictionary later
    static const std::shared_ptr<FvccGeometryScheme> readOrCreate(const UnstructuredMesh& uMesh);

private:

    const Executor exec_;
    const UnstructuredMesh& uMesh_;
    std::unique_ptr<FvccGeometrySchemeKernel> kernel_;

    fvcc::SurfaceField<scalar> weights_;
    fvcc::SurfaceField<scalar> deltaCoeffs_;
    fvcc::SurfaceField<scalar> nonOrthDeltaCoeffs_;
    fvcc::SurfaceField<Vector> nonOrthCorrectionVectors_;
};


} // namespace NeoFOAM
