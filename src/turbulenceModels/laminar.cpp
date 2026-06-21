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
    , gradOp_(rt.exec, rt.nfMesh)
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
{}

void Laminar::validate(const nnfvcc::VolumeField<NeoN::Vec3>& U)
{
    gradOp_.gradTensor(U, gradU_);
    gradU_.correctBoundaryConditions();
    surfInterp_.interpolate(nu_, nuEff_);
}

void Laminar::
    correct(const nnfvcc::VolumeField<NeoN::Vec3>& U, nnfvcc::SurfaceField<NeoN::scalar>&, RunTime&)
{
    gradOp_.gradTensor(U, gradU_);
    gradU_.correctBoundaryConditions();
    surfInterp_.interpolate(nu_, nuEff_);
}

nnfvcc::SurfaceField<NeoN::scalar>& Laminar::nuEff() { return nuEff_; }

const nnfvcc::VolumeField<NeoN::scalar>& Laminar::nut() const { return nut_; }

const nnfvcc::VolumeField<NeoN::Tensor>& Laminar::gradU() const { return gradU_; }

} // namespace NeoFOAM
