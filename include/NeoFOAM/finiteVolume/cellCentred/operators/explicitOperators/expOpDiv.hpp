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

class DivEqnTerm : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    DivEqnTerm(
        const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux,
        fvcc::VolumeField<NeoFOAM::scalar>& Phi,
        const Input& input
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true),
          termType_(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit), exec_(Phi.exec()),
          nCells_(Phi.mesh().nCells()), faceFlux_(faceFlux), Phi_(Phi),
          div_(Phi.exec(), Phi.mesh(), input)
    {}

    DivEqnTerm(
        const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& Phi
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(false),
          termType_(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit), exec_(Phi.exec()),
          nCells_(Phi.mesh().nCells()), faceFlux_(faceFlux), Phi_(Phi), div_(Phi.exec(), Phi.mesh())
    {}

    void build(const NeoFOAM::Input& input)
    {
        if (std::holds_alternative<NeoFOAM::Dictionary>(input))
        {
            auto dict = std::get<NeoFOAM::Dictionary>(input);
            std::string schemeName = "div(" + faceFlux_.name + "," + Phi_.name + ")";
            auto tokens = dict.subDict("divSchemes").get<NeoFOAM::TokenList>(schemeName);
            div_.build(tokens);
        }
        else
        {
            auto tokens = std::get<NeoFOAM::TokenList>(input);
            div_.build(tokens);
        }
    }

    std::string display() const { return "DivEqnTerm"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        NeoFOAM::Field<NeoFOAM::scalar> tmpsource(source.exec(), source.size(), 0.0);
        div_.div(tmpsource, faceFlux_, Phi_);
        source += tmpsource;
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
