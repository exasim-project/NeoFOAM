// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"

class Laplacian : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    Laplacian(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells,
        NeoFOAM::scalar value
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true), termType_(termType), exec_(exec),
          nCells_(nCells), value(value)
    {}

    Laplacian(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(false), termType_(termType), exec_(exec),
          nCells_(nCells), value(1.0)
    {}

    // constructor for Laplacian from Input
    Laplacian(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells,
        const NeoFOAM::Input& input
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true), termType_(termType), exec_(exec),
          nCells_(nCells), value(read(input))
    {}

    NeoFOAM::scalar read(const NeoFOAM::Input& input)
    {
        NeoFOAM::scalar value = 0.0;
        if (std::holds_alternative<NeoFOAM::Dictionary>(input))
        {
            value = std::get<NeoFOAM::Dictionary>(input).get<NeoFOAM::scalar>("value");
        }
        else
        {
            value = std::get<NeoFOAM::TokenList>(input).get<NeoFOAM::scalar>(0);
        }
        return value;
    }

    void build(const NeoFOAM::Input& input)
    {
        value = read(input);
        termEvaluated = true;
    }

    std::string display() const { return "Laplacian"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        NeoFOAM::scalar setValue = value;
        auto scale = scaleField();
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale[i] * setValue; }
        );
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;


    const NeoFOAM::Executor exec_;
    std::size_t nCells_;
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
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true), termType_(termType), exec_(exec),
          nCells_(nCells), value(value)
    {}

    Divergence(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(false), termType_(termType), exec_(exec),
          nCells_(nCells), value(1.0)
    {}

    // constructor for Divergence from Input
    Divergence(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells,
        const NeoFOAM::Input& input
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true), termType_(termType), exec_(exec),
          nCells_(nCells), value(read(input))
    {}

    void build(const NeoFOAM::Input& input)
    {
        value = read(input);
        termEvaluated = true;
    }

    NeoFOAM::scalar read(const NeoFOAM::Input& input)
    {
        NeoFOAM::scalar value = 0.0;
        if (std::holds_alternative<NeoFOAM::Dictionary>(input))
        {
            value = std::get<NeoFOAM::Dictionary>(input).get<NeoFOAM::scalar>("value");
        }
        else
        {
            value = std::get<NeoFOAM::TokenList>(input).get<NeoFOAM::scalar>(0);
        }
        return value;
    }

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        NeoFOAM::scalar setValue = value;
        auto scale = scaleField();
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale[i] * setValue; }
        );
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;


    const NeoFOAM::Executor exec_;
    std::size_t nCells_;
    NeoFOAM::scalar value = 1.0;
};
