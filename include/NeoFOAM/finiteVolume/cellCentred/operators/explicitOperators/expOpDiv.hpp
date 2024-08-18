// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace dsl = NeoFOAM::DSL;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::finiteVolume::cellCentred::expOp
{

class DivScheme
{

public:

    DivScheme(
        const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux,
        fvcc::VolumeField<NeoFOAM::scalar>& Phi,
        const Input& input
    )
        : termType_(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit), exec_(Phi.exec()),
          nCells_(Phi.mesh().nCells()), faceFlux_(faceFlux), Phi_(Phi),
          div_(Phi.exec(), Phi.mesh(), input)
    {}

    std::string display() const { return "DivScheme"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        div_.div(source, faceFlux_, Phi_);
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return &Phi_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;

    const NeoFOAM::Executor exec_;
    const std::size_t nCells_;
    const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux_;
    fvcc::VolumeField<NeoFOAM::scalar>& Phi_;
    fvcc::DivOperator div_;
};


} // namespace NeoFOAM
