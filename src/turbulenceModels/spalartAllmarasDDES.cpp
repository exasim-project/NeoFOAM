// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/spalartAllmarasDDES.hpp"

namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

SpalartAllmarasDDES::SpalartAllmarasDDES(
    const NeoN::Executor& exec,
    const NeoN::UnstructuredMesh& mesh,
    const nnfvcc::VolumeField<scalar>& nu,
    const nnfvcc::VolumeField<scalar>& wallDist,
    const nnfvcc::VolumeField<scalar>& nearWallDist,
    const nnfvcc::VolumeField<scalar>& delta
)
    : exec_(exec)
    , mesh_(mesh)
    , nu_(nu)
    , wallDist_(wallDist)
    , nearWallDist_(nearWallDist)
    , delta_(delta)
    , surfNu_(
          exec,
          "surfNu",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradU_(
          exec,
          "gradU",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<NeoN::Tensor>>(mesh)
      )
    , gradNuTilda_(
          exec,
          "gradNuTilda",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh)
      )
    , magSqrGradNuTilda_(
          exec,
          "magSqrGradNuTilda",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , production_(
          exec,
          "production",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , spCoeff_(
          exec,
          "spCoeff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNut_(
          exec,
          "surfNut",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , surfNuTilda_(
          exec,
          "surfNuTilda",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , nuEff_(exec, "nuEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , nuTildaEff_(
          exec,
          "nuTildaEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradOp_(exec, mesh)
    , physicsModel_(exec, mesh)
{
    // Pre-compute face-interpolated nu (constant in time for single-phase incompressible flows)
    fvcc::SurfaceInterpolation<scalar> surfInterp(
        exec_,
        mesh_,
        NeoN::TokenList({std::string("linear")})
    );
    surfInterp.interpolate(nu_, surfNu_);
}

void SpalartAllmarasDDES::validate(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::VolumeField<scalar>& nuTilda,
    nnfvcc::VolumeField<scalar>& nut
)
{
    gradOp_.gradTensor(U, gradU_);
    physicsModel_.calcNuTildaDiffusionCoeff(nuTilda, surfNu_, surfNuTilda_, nuTildaEff_);
    physicsModel_.correctNut(nut, surfNut_, nuEff_, nuTilda, nu_, surfNu_, U, nearWallDist_);
}

void SpalartAllmarasDDES::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    nnfvcc::VolumeField<scalar>& nuTilda,
    nnfvcc::VolumeField<scalar>& nut,
    RunTime& rt
)
{
    // 1. Recompute velocity and nuTilda gradients
    gradOp_.gradTensor(U, gradU_);
    gradOp_.grad(nuTilda, gradNuTilda_);
    physicsModel_.calcMagSqrVec(magSqrGradNuTilda_, gradNuTilda_);

    // 2. Compute SA-DDES production and destruction source terms
    physicsModel_.computeProdSpDDES(
        production_,
        spCoeff_,
        nuTilda,
        nu_,
        gradU_,
        wallDist_,
        delta_,
        magSqrGradNuTilda_
    );

    // 3. Solve nuTilda transport equation
    PDESolver<scalar> nuTildaEqn(
        dsl::imp::ddt(nuTilda) + dsl::imp::div(phi, nuTilda)
            - dsl::imp::laplacian(nuTildaEff_, nuTilda) + dsl::imp::source(spCoeff_, nuTilda)
            - dsl::exp::source(production_),
        nuTilda,
        rt
    );
    nuTildaEqn.solve();

    // 4. Update derived quantities
    physicsModel_.calcNuTildaDiffusionCoeff(nuTilda, surfNu_, surfNuTilda_, nuTildaEff_);
    physicsModel_.correctNut(nut, surfNut_, nuEff_, nuTilda, nu_, surfNu_, U, nearWallDist_);
}

nnfvcc::SurfaceField<scalar>& SpalartAllmarasDDES::nuEff() { return nuEff_; }

nnfvcc::SurfaceField<scalar>& SpalartAllmarasDDES::nuTildaEff() { return nuTildaEff_; }

const nnfvcc::VolumeField<NeoN::Tensor>& SpalartAllmarasDDES::gradU() const { return gradU_; }

} // namespace NeoFOAM
