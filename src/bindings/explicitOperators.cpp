// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Bindings for the cross-backend operator parity tests (test/operators/):
//   - evaluate_explicit: zero the result vector, resolve the operator's
//     discretisation scheme (from the mapped fvSchemes Dictionary or a raw
//     TokenList), and run its explicitOperation — the Python equivalent of
//     what test/operators.cpp does with read() + explicitOperation().
//   - exp_div: dsl::exp::div(flux, VolumeField<Vec3>) — the _neon module
//     only binds the scalar explicit div overloads.
//   - evaluate_implicit: assemble the operator into a fresh linear system and
//     return the matrix-vector residual A·x − b, the volume-integrated
//     equivalent of the explicit operation (cf. applyOperator/operator& in
//     include/NeoFOAM/datastructures/pde.hpp).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/dsl/explicit.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"

#include "bindings.hpp"

// Same registration workaround as NeoN's src/bindings/dsl.cpp (see the TODO
// there): the operators are extern templates whose factory self-registration
// only fires in libNeoN, while this hidden-visibility module holds its own
// empty copy of the factory lookup tables. Operators constructed in this TU
// (exp_div) resolve their scheme against this module's tables, so instantiate
// them here to run the self-registration inside neofoam_bindings.
namespace NeoN::finiteVolume::cellCentred
{
template class GaussGreenDiv<scalar>;
template class GaussGreenDiv<Vec3>;
template class GaussGreenDiv<Vec3, scalar>;
template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vec3>;
template class GaussGreenLaplacian<Vec3, scalar>;
} // namespace NeoN::finiteVolume::cellCentred

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace
{

template<typename ValueType, typename SchemeType>
void evaluateExplicit(
    NeoN::dsl::SpatialOperator<ValueType>& op,
    const SchemeType& schemes,
    NeoN::Vector<ValueType>& result
)
{
    // explicitOperation accumulates into the source vector, so start from zero.
    NeoN::fill(result, NeoN::zero<ValueType>());
    op.read(NeoN::Input {schemes});
    op.explicitOperation(result);
}

template<typename ValueType, typename SchemeType>
void evaluateImplicit(
    NeoN::dsl::SpatialOperator<ValueType>& op,
    const SchemeType& schemes,
    const fvcc::VolumeField<ValueType>& psi,
    NeoN::Vector<ValueType>& result
)
{
    // Segregated scalar-matrix system — the only form computeResidual is
    // instantiated for; for scalar fields it is the plain LinearSystem<scalar>.
    // The system is freshly created, so the accumulation of implicitOperation
    // starts from zero.
    auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar, ValueType>(psi.mesh());
    op.read(NeoN::Input {schemes});
    op.implicitOperation(ls);
    // A·x − b: the volume-integrated equivalent of the explicit operation.
    NeoN::la::computeResidual(ls.matrix(), ls.rhs(), psi.internalVector(), result);
}

} // namespace

namespace NeoFOAM::bindings
{

void registerExplicitOperators(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Explicit vector convection div(phi, U) — scalar overloads live in _neon.
    // The operator stores references to flux and field; keep them alive.
    // -------------------------------------------------------------------
    m.def(
        "exp_div",
        [](const fvcc::SurfaceField<NeoN::scalar>& faceFlux, fvcc::VolumeField<NeoN::Vec3>& field)
        { return NeoN::dsl::exp::div(faceFlux, field); },
        "face_flux"_a,
        "field"_a,
        nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 2>(),
        "Explicit vector divergence operator div(phi, U)"
    );

    // -------------------------------------------------------------------
    // Evaluate an explicit spatial operator into a result vector:
    // zero result, read the scheme (fvSchemes Dictionary or TokenList),
    // run explicitOperation. Overloads: {scalar, Vec3} x {Dictionary, TokenList}.
    // -------------------------------------------------------------------
    m.def(
        "evaluate_explicit",
        &evaluateExplicit<NeoN::scalar, NeoN::Dictionary>,
        "op"_a,
        "schemes"_a,
        "result"_a,
        "Evaluate an explicit scalar operator; scheme looked up in the fvSchemes dict"
    );
    m.def(
        "evaluate_explicit",
        &evaluateExplicit<NeoN::Vec3, NeoN::Dictionary>,
        "op"_a,
        "schemes"_a,
        "result"_a,
        "Evaluate an explicit vector operator; scheme looked up in the fvSchemes dict"
    );
    m.def(
        "evaluate_explicit",
        &evaluateExplicit<NeoN::scalar, NeoN::TokenList>,
        "op"_a,
        "schemes"_a,
        "result"_a,
        "Evaluate an explicit scalar operator with raw scheme tokens"
    );
    m.def(
        "evaluate_explicit",
        &evaluateExplicit<NeoN::Vec3, NeoN::TokenList>,
        "op"_a,
        "schemes"_a,
        "result"_a,
        "Evaluate an explicit vector operator with raw scheme tokens"
    );

    // -------------------------------------------------------------------
    // Evaluate an implicit spatial operator as a matrix-vector product:
    // assemble A and b into a fresh linear system, then result = A·psi − b.
    // The result is volume-integrated — divide by the cell volumes to get
    // the per-volume density the explicit operators produce.
    // Overloads: {scalar, Vec3} x {Dictionary, TokenList}.
    // -------------------------------------------------------------------
    m.def(
        "evaluate_implicit",
        &evaluateImplicit<NeoN::scalar, NeoN::Dictionary>,
        "op"_a,
        "schemes"_a,
        "psi"_a,
        "result"_a,
        "Apply an implicit scalar operator (A·psi − b); scheme from the fvSchemes dict"
    );
    m.def(
        "evaluate_implicit",
        &evaluateImplicit<NeoN::Vec3, NeoN::Dictionary>,
        "op"_a,
        "schemes"_a,
        "psi"_a,
        "result"_a,
        "Apply an implicit vector operator (A·psi − b); scheme from the fvSchemes dict"
    );
    m.def(
        "evaluate_implicit",
        &evaluateImplicit<NeoN::scalar, NeoN::TokenList>,
        "op"_a,
        "schemes"_a,
        "psi"_a,
        "result"_a,
        "Apply an implicit scalar operator (A·psi − b) with raw scheme tokens"
    );
    m.def(
        "evaluate_implicit",
        &evaluateImplicit<NeoN::Vec3, NeoN::TokenList>,
        "op"_a,
        "schemes"_a,
        "psi"_a,
        "result"_a,
        "Apply an implicit vector operator (A·psi − b) with raw scheme tokens"
    );
}

} // namespace NeoFOAM::bindings
