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


class DdtScheme : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    DdtScheme(fvcc::VolumeField<NeoFOAM::scalar>& Phi)
        : dsl::EqnTermMixin<NeoFOAM::scalar>(),
          termType_(dsl::EqnTerm<NeoFOAM::scalar>::Type::Temporal), Phi_(Phi), exec_(Phi.exec()),
          nCells_(Phi.mesh().nCells())
    {}

    std::string display() const { return "DdtScheme"; }

    void temporalOperation(NeoFOAM::Field<NeoFOAM::scalar>& field, NeoFOAM::scalar scale) {}

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return &Phi_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;


    fvcc::VolumeField<NeoFOAM::scalar>& Phi_;
    const NeoFOAM::Executor exec_;
    const std::size_t nCells_;
};

} // namespace NeoFOAM
