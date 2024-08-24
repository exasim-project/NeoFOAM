// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"


#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"


namespace dsl = NeoFOAM::DSL;

class Divergence : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    Divergence(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true), termType_(termType), exec_(exec),
          nCells_(nCells)
    {}

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        auto scale = scaleValue();
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1.0 * scale; }
        );
    }

    void build(const NeoFOAM::Input& input)
    {
        // do nothing
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() const { return nullptr; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;
    NeoFOAM::Executor exec_;
    std::size_t nCells_;
};

class TimeTerm : public dsl::EqnTermMixin<NeoFOAM::scalar>
{

public:

    TimeTerm(
        dsl::EqnTerm<NeoFOAM::scalar>::Type termType,
        const NeoFOAM::Executor& exec,
        std::size_t nCells
    )
        : dsl::EqnTermMixin<NeoFOAM::scalar>(true), termType_(termType), exec_(exec),
          nCells_(nCells)
    {}

    std::string display() const { return "TimeTerm"; }

    void build(const NeoFOAM::Input& input)
    {
        // do nothing
    }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        auto scale = scaleValue();
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1 * scale; }
        );
    }

    dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() const { return nullptr; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;
    const NeoFOAM::Executor exec_;
    std::size_t nCells_;
};


TEST_CASE("TimeIntegration")
{
    auto exec = NeoFOAM::SerialExecutor();
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Dictionary dict;
    dict.insert("type", std::string("forwardEuler"));

    dsl::EqnTerm<NeoFOAM::scalar> divTerm =
        Divergence(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, 1);

    dsl::EqnTerm<NeoFOAM::scalar> ddtTerm =
        TimeTerm(dsl::EqnTerm<NeoFOAM::scalar>::Type::Temporal, exec, 1);

    dsl::EqnSystem eqnSys = ddtTerm + divTerm;

    fvcc::TimeIntegration timeIntergrator(eqnSys, dict);
}
