// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/stencil/basicFvccGeometryScheme.hpp"
#include "NeoFOAM/mesh/stencil/fvccGeometryScheme.hpp"
#include "NeoFOAM/core/error.hpp"
#include <any>

namespace NeoFOAM
{

FvccGeometrySchemeKernel::FvccGeometrySchemeKernel(const UnstructuredMesh& uMesh) {}


const std::shared_ptr<FvccGeometryScheme>
FvccGeometryScheme::readOrCreate(const UnstructuredMesh& uMesh)
{
    StencilDataBase& stencil_db = uMesh.stencilDB();
    if (!stencil_db.contains("FvccGeometryScheme"))
    {
        stencil_db.insert(
            std::string("FvccGeometryScheme"), std::make_shared<FvccGeometryScheme>(uMesh)
        );
    }
    return stencil_db.get<std::shared_ptr<FvccGeometryScheme>>("FvccGeometryScheme");
}


FvccGeometryScheme::FvccGeometryScheme(
    const Executor& exec,
    std::unique_ptr<FvccGeometrySchemeKernel> kernel,
    const fvcc::SurfaceField<scalar>& weights,
    const fvcc::SurfaceField<scalar>& deltaCoeffs,
    const fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs,
    const fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
    : exec_(exec), uMesh_(weights.mesh()), kernel_(std::move(kernel)), weights_(weights),
      deltaCoeffs_(deltaCoeffs), nonOrthDeltaCoeffs_(nonOrthDeltaCoeffs),
      nonOrthCorrectionVectors_(nonOrthCorrectionVectors)
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
}

FvccGeometryScheme::FvccGeometryScheme(
    const Executor& exec,
    const UnstructuredMesh& uMesh,
    std::unique_ptr<FvccGeometrySchemeKernel> kernel
)
    : exec_(exec), uMesh_(uMesh), kernel_(std::move(kernel)),
      weights_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
      deltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
      nonOrthDeltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
      nonOrthCorrectionVectors_(uMesh.exec(), uMesh, createCalculatedBCs<Vector>(uMesh))
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
    update();
}

FvccGeometryScheme::FvccGeometryScheme(const UnstructuredMesh& uMesh)
    : exec_(uMesh.exec()), uMesh_(uMesh),
      kernel_(std::make_unique<NeoFOAM::BasicFvccGeometryScheme>(uMesh)
      ), // TODO add selection mechanism
      weights_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
      deltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
      nonOrthDeltaCoeffs_(uMesh.exec(), uMesh, createCalculatedBCs<scalar>(uMesh)),
      nonOrthCorrectionVectors_(uMesh.exec(), uMesh, createCalculatedBCs<Vector>(uMesh))
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
    update();
}

std::string FvccGeometryScheme::name() const { return std::string("FvccGeometryScheme"); }

void FvccGeometryScheme::update()
{
    std::visit(
        [&](const auto& exec)
        {
            kernel_->updateWeights(exec, weights_);
            // kernel_->updateDeltaCoeffs(exec, deltaCoeffs_);
            // kernel_->updateNonOrthDeltaCoeffs(exec, nonOrthDeltaCoeffs_);
            // kernel_->updateNonOrthCorrectionVectors(exec, nonOrthCorrectionVectors_);
        },
        exec_
    );
}

const fvcc::SurfaceField<scalar>& FvccGeometryScheme::weights() const { return weights_; }

const fvcc::SurfaceField<scalar>& FvccGeometryScheme::deltaCoeffs() const { return deltaCoeffs_; }

const fvcc::SurfaceField<scalar>& FvccGeometryScheme::nonOrthDeltaCoeffs() const
{
    return nonOrthDeltaCoeffs_;
}

const fvcc::SurfaceField<Vector>& FvccGeometryScheme::nonOrthCorrectionVectors() const
{
    return nonOrthCorrectionVectors_;
}

} // namespace NeoFOAM
