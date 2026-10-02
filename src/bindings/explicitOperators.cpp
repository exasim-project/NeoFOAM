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
//   - GradScheme: the gradient operator the case's gradSchemes selects for one
//     field (e.g. "grad(U) cellLimited Gauss linear 1"), as a reusable handle
//     over the explicit tensor gradient the viscous stress consumes.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <string>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/dsl/explicit.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/boundedDiv.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/cellLimitedGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/corrected.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/limitedCorrected.hpp"

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"

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
// OpenFOAM's "bounded <scheme>" convection wrapper; without it a case whose divSchemes
// carry the prefix aborts here with "Could not find constructor for bounded".
template class BoundedDiv<scalar>;
template class BoundedDiv<Vec3>;
template class BoundedDiv<Vec3, scalar>;
template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vec3>;
template class GaussGreenLaplacian<Vec3, scalar>;
// Carries the halo exchange of the tensor gradient field GradScheme allocates in
// this TU; NeoN's boundary.hpp instantiates the calculated boundary for Tensor
// but not the processor one.
template class volumeBoundary::Processor<Tensor>;
} // namespace NeoN::finiteVolume::cellCentred

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace
{

// Gradient schemes are plain classes, so the workaround above has no template to
// instantiate: odr-use their registration flags instead to run the same
// self-registration against this module's factory table.
[[maybe_unused]] const bool* gaussGreenGradRegistration = &fvcc::GaussGreenGrad::REGISTERED;
[[maybe_unused]] const bool* cellLimitedGradRegistration = &fvcc::CellLimitedGrad::REGISTERED;

/* @brief The gradient operator a case selects for one field in gradSchemes.
 *
 * NeoN looks a gradient scheme up under its literal key ("grad(U)") with no
 * fall-through to "default", so a case that does not name the field keeps the
 * plain Gauss-Green gradient this replaced.
 */
class GradScheme
{
public:

    GradScheme(nf::RunTime& rt, const std::string& fieldName)
        : grad_(fvcc::GradOperatorFactory<NeoN::Vec3>::create(
            rt.exec,
            rt.nfMesh,
            readScheme(rt.fvSchemesDict, fieldName)
        ))
    {}

    /* @brief The tensor gradient grad(u) of a vector field, e.g. grad(U). */
    fvcc::VolumeField<NeoN::Tensor> gradTensor(const fvcc::VolumeField<NeoN::Vec3>& u) const
    {
        // Proc-aware calculated BCs as in GaussGreenGrad::gradTensor: processor
        // patches need the halo-exchange BC to hold the neighbour value.
        auto bcs = fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(u.mesh());
        fvcc::VolumeField<NeoN::Tensor> gradU(u.exec(), "gradU", u.mesh(), bcs);
        grad_->gradTensor(u, gradU, NeoN::dsl::Coeff {});
        return gradU;
    }

private:

    static NeoN::Input readScheme(const NeoN::Dictionary& schemes, const std::string& fieldName)
    {
        const std::string key = "grad(" + fieldName + ")";
        if (schemes.contains("gradSchemes"))
        {
            const NeoN::Dictionary& gradSchemes = schemes.subDict("gradSchemes");
            // A multi-word scheme ("Gauss linear") converts to a TokenList, a
            // single-word one ("pointCellsLeastSquares") to a plain string.
            if (gradSchemes.isType<NeoN::TokenList>(key))
            {
                return gradSchemes.get<NeoN::TokenList>(key);
            }
            if (gradSchemes.isType<std::string>(key))
            {
                return NeoN::TokenList({gradSchemes.get<std::string>(key)});
            }
        }
        return NeoN::TokenList({std::string("Gauss"), std::string("linear")});
    }

    std::unique_ptr<fvcc::GradOperatorFactory<NeoN::Vec3>> grad_;
};

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

    // -------------------------------------------------------------------
    // Runtime-selected gradient operator: reads gradSchemes/grad(<field>) —
    // Gauss, cellLimited, ... — and computes the explicit tensor gradient.
    // -------------------------------------------------------------------
    nb::class_<GradScheme>(m, "GradScheme")
        .def(
            nb::init<nf::RunTime&, const std::string&>(),
            "runtime"_a,
            "field"_a,
            nb::keep_alive<1, 2>(), // the mesh reference must outlive the operator
            "Construct the gradient operator the case's gradSchemes selects for grad(<field>)"
        )
        .def(
            "grad_tensor",
            &GradScheme::gradTensor,
            "u"_a,
            "Compute the velocity gradient tensor field grad(U)"
        );
}

} // namespace NeoFOAM::bindings
