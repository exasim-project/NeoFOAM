// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"
#include "NeoFOAM/DSL/eqnTermBuilder.hpp"

class Laplacian : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    Laplacian(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells,
        NeoFOAM::scalar value
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(), termType_(termType), exec_(exec), nCells_(nCells),
          value(value)
    {}

    std::string display() const { return "Laplacian"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        NeoFOAM::scalar value2 = value;
        if (scaleField())
        {
            auto sField = scaleField()->span();
            NeoFOAM::scalar scale = scaleValue();
            auto sourceField = source.span();
            NeoFOAM::parallelFor(
                source.exec(),
                {0, source.size()},
                KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale * sField[i] * value2; }
            );
        }
        else
        {
            NeoFOAM::scalar scale = scaleValue();
            auto sourceField = source.span();
            NeoFOAM::parallelFor(
                source.exec(),
                {0, source.size()},
                KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale * value2; }
            );
        }
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    const std::size_t nCells() const { return nCells_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;


    const NeoFOAM::Executor exec_;
    const std::size_t nCells_;
    NeoFOAM::scalar value = 1.0;
};


class Divergence : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    Divergence(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells,
        NeoFOAM::scalar value
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(), termType_(termType), exec_(exec), nCells_(nCells),
          value(value)
    {}

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        NeoFOAM::scalar value2 = value;
        if (scaleField())
        {
            auto sField = scaleField()->span();
            NeoFOAM::scalar scale = scaleValue();
            auto sourceField = source.span();
            NeoFOAM::parallelFor(
                source.exec(),
                {0, source.size()},
                KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale * sField[i] * value2; }
            );
        }
        else
        {
            NeoFOAM::scalar scale = scaleValue();
            auto sourceField = source.span();
            NeoFOAM::parallelFor(
                source.exec(),
                {0, source.size()},
                KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale * value2; }
            );
        }
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    const std::size_t nCells() const { return nCells_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;


    const NeoFOAM::Executor exec_;
    const std::size_t nCells_;
    NeoFOAM::scalar value = 1.0;
};
