// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/turbulenceModels/laminar.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

Laminar::Laminar(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu)
    : exec_(rt.exec)
    , mesh_(rt.nfMesh)
    , nu_(nu)
    , gradUOp_(gradSchemePtr(rt, "grad(U)"))
    , gradU_(
          rt.exec,
          "gradU",
          rt.nfMesh,
          fvcc::createCalculatedProcBCs<nnfvcc::VolumeBoundary<NeoN::Tensor>>(rt.nfMesh)
      )
    , nuEff_(
          rt.exec,
          "nuEff",
          rt.nfMesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh)
      )
    , nut_(
          rt.exec,
          "nut",
          rt.nfMesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh)
      )
    , surfInterp_(rt.exec, rt.nfMesh, NeoN::TokenList({std::string("linear")}))
{
    // nut is identically zero for the laminar model, but the VolumeField constructor
    // above only ALLOCATES nut_ (from the Umpire pool, which does not zero memory) — it
    // never initialises it. viscousStress() reads nuEff = nu + nut per cell, so an
    // uninitialised nut_ injects garbage into the momentum source and diverges the solve.
    // Pin it to zero here (internal + boundary) so nuEff == nu as documented.
    NeoN::fill(nut_.internalVector(), NeoN::scalar(0));
    NeoN::fill(nut_.boundaryData().value(), NeoN::scalar(0));
}

void Laminar::updateGradU(const nnfvcc::VolumeField<NeoN::Vec3>& U)
{
    gradUOp_->gradTensor(U, gradU_, NeoN::dsl::Coeff {});
    gradU_.correctBoundaryConditions();
}

void Laminar::validate(const nnfvcc::VolumeField<NeoN::Vec3>& U)
{
    updateGradU(U);
    surfInterp_.interpolate(nu_, nuEff_);
}

void Laminar::
    correct(const nnfvcc::VolumeField<NeoN::Vec3>& U, nnfvcc::SurfaceField<NeoN::scalar>&, RunTime&)
{
    updateGradU(U);
    surfInterp_.interpolate(nu_, nuEff_);
}

nnfvcc::SurfaceField<NeoN::scalar>& Laminar::nuEff() { return nuEff_; }

const nnfvcc::VolumeField<NeoN::scalar>& Laminar::nut() const { return nut_; }

const nnfvcc::VolumeField<NeoN::Tensor>& Laminar::gradU() const { return gradU_; }

} // namespace NeoFOAM
