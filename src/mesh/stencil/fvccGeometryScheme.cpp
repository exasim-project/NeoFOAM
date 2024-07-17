// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/stencil/basicFvccGeometryScheme.hpp"
#include "NeoFOAM/mesh/stencil/fvccGeometryScheme.hpp"
#include "NeoFOAM/core/error.hpp"
#include <any>

namespace NeoFOAM
{
namespace fvcc = finiteVolume::cellCentred;

GeometrySchemeFactory::GeometrySchemeFactory(const UnstructuredMesh& mesh) {}


const std::shared_ptr<GeometryScheme> GeometryScheme::readOrCreate(const UnstructuredMesh& mesh)
{
    StencilDataBase& stencil_db = mesh.stencilDB();
    if (!stencil_db.contains("GeometryScheme"))
    {
        stencil_db.insert(std::string("GeometryScheme"), std::make_shared<GeometryScheme>(mesh));
    }
    return stencil_db.get<std::shared_ptr<GeometryScheme>>("GeometryScheme");
}


GeometryScheme::GeometryScheme(
    const Executor& exec,
    std::unique_ptr<GeometrySchemeFactory> kernel,
    const fvcc::SurfaceField<scalar>& weights,
    const fvcc::SurfaceField<scalar>& deltaCoeffs,
    const fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs,
    const fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
    : exec_(exec), mesh_(weights.mesh()), kernel_(std::move(kernel)), weights_(weights),
      deltaCoeffs_(deltaCoeffs), nonOrthDeltaCoeffs_(nonOrthDeltaCoeffs),
      nonOrthCorrectionVectors_(nonOrthCorrectionVectors)
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
}

GeometryScheme::GeometryScheme(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    std::unique_ptr<GeometrySchemeFactory> kernel
)
    : exec_(exec), mesh_(mesh), kernel_(std::move(kernel)),
      weights_(mesh.exec(), mesh, fvcc::createCalculatedBCs<scalar>(mesh)),
      deltaCoeffs_(mesh.exec(), mesh, fvcc::createCalculatedBCs<scalar>(mesh)),
      nonOrthDeltaCoeffs_(mesh.exec(), mesh, fvcc::createCalculatedBCs<scalar>(mesh)),
      nonOrthCorrectionVectors_(mesh.exec(), mesh, fvcc::createCalculatedBCs<Vector>(mesh))
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
    update();
}

GeometryScheme::GeometryScheme(const UnstructuredMesh& mesh)
    : exec_(mesh.exec()), mesh_(mesh),
      kernel_(std::make_unique<NeoFOAM::BasicGeometryScheme>(mesh)), // TODO add selection mechanism
      weights_(mesh.exec(), mesh, fvcc::createCalculatedBCs<scalar>(mesh)),
      deltaCoeffs_(mesh.exec(), mesh, fvcc::createCalculatedBCs<scalar>(mesh)),
      nonOrthDeltaCoeffs_(mesh.exec(), mesh, fvcc::createCalculatedBCs<scalar>(mesh)),
      nonOrthCorrectionVectors_(mesh.exec(), mesh, fvcc::createCalculatedBCs<Vector>(mesh))
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
    update();
}

std::string GeometryScheme::name() const { return std::string("GeometryScheme"); }

void GeometryScheme::update()
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

const fvcc::SurfaceField<scalar>& GeometryScheme::weights() const { return weights_; }

const fvcc::SurfaceField<scalar>& GeometryScheme::deltaCoeffs() const { return deltaCoeffs_; }

const fvcc::SurfaceField<scalar>& GeometryScheme::nonOrthDeltaCoeffs() const
{
    return nonOrthDeltaCoeffs_;
}

const fvcc::SurfaceField<Vector>& GeometryScheme::nonOrthCorrectionVectors() const
{
    return nonOrthCorrectionVectors_;
}

} // namespace NeoFOAM
