// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/stencil/FvccGeometryScheme.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryFieldSelector.hpp"
#include "NeoFOAM/core/Error.hpp"
#include "NeoFOAM/mesh/stencil/BasicFvccGeometryScheme.hpp"
#include <any>
namespace NeoFOAM
{

FvccGeometrySchemeKernel::FvccGeometrySchemeKernel(
    const unstructuredMesh& uMesh
)
{
}


const std::shared_ptr<FvccGeometryScheme> FvccGeometryScheme::readOrCreate(const unstructuredMesh& uMesh)
{
    StencilDataBase& stencil_db = uMesh.stencilDB();
    if (!stencil_db.contains("FvccGeometryScheme"))
    {
        stencil_db.insert(std::string("FvccGeometryScheme"), std::make_shared<FvccGeometryScheme>(uMesh));
    }
    return stencil_db.get<std::shared_ptr<FvccGeometryScheme>>("FvccGeometryScheme");
}


FvccGeometryScheme::FvccGeometryScheme(
    const executor& exec,
    std::unique_ptr<FvccGeometrySchemeKernel> kernel,
    const fvccSurfaceField<scalar>& weights,
    const fvccSurfaceField<scalar>& deltaCoeffs,
    const fvccSurfaceField<scalar>& nonOrthDeltaCoeffs,
    const fvccSurfaceField<Vector>& nonOrthCorrectionVectors
) : exec_(exec),
    uMesh_(weights.mesh()),
    kernel_(std::move(kernel)),
    weights_(weights),
    deltaCoeffs_(deltaCoeffs),
    nonOrthDeltaCoeffs_(nonOrthDeltaCoeffs),
    nonOrthCorrectionVectors_(nonOrthCorrectionVectors)
{
    if (kernel_ == nullptr)
    {
        error("Kernel is not initialized").exit();
    }
}

FvccGeometryScheme::FvccGeometryScheme(
    const executor& exec,
    const unstructuredMesh& uMesh,
    std::unique_ptr<FvccGeometrySchemeKernel> kernel
) : exec_(exec),
    uMesh_(uMesh),
    kernel_(std::move(kernel)),
    weights_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
    deltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
    nonOrthDeltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
    nonOrthCorrectionVectors_(uMesh.exec(), uMesh, createCalculatedBCs<Vector>(uMesh))
{
    if (kernel_ == nullptr)
    {
        error("Kernel is not initialized").exit();
    }
    update();
}

FvccGeometryScheme::FvccGeometryScheme(
    const unstructuredMesh& uMesh
) : exec_(uMesh.exec()),
    uMesh_(uMesh),
    kernel_(std::make_unique<NeoFOAM::BasicFvccGeometryScheme>(uMesh)), // TODO add selection mechanism
    weights_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
    deltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
    nonOrthDeltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
    nonOrthCorrectionVectors_(uMesh.exec(), uMesh, createCalculatedBCs<Vector>(uMesh))
{
    if (kernel_ == nullptr)
    {
        error("Kernel is not initialized").exit();
    }
    update();
}

std::string FvccGeometryScheme::name() const
{
    return std::string("FvccGeometryScheme");
}

void FvccGeometryScheme::update()
{
    std::visit([&](const auto& exec)
               {
                   kernel_->updateWeights(exec, weights_);
                   // kernel_->updateDeltaCoeffs(exec, deltaCoeffs_);
                   // kernel_->updateNonOrthDeltaCoeffs(exec, nonOrthDeltaCoeffs_);
                   // kernel_->updateNonOrthCorrectionVectors(exec, nonOrthCorrectionVectors_);
               },
               exec_);
}

const fvccSurfaceField<scalar>& FvccGeometryScheme::weights() const
{
    return weights_;
}

const fvccSurfaceField<scalar>& FvccGeometryScheme::deltaCoeffs() const
{
    return deltaCoeffs_;
}

const fvccSurfaceField<scalar>& FvccGeometryScheme::nonOrthDeltaCoeffs() const
{
    return nonOrthDeltaCoeffs_;
}

const fvccSurfaceField<Vector>& FvccGeometryScheme::nonOrthCorrectionVectors() const
{
    return nonOrthCorrectionVectors_;
}

} // namespace NeoFOAM
