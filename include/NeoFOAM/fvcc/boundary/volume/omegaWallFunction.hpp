// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail
{

// Default coefficients matching OpenFOAM's wallFunctionCoefficients defaults
// (src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
//  wallFunction/wallFunctionCoefficients/wallFunctionCoefficients.C:61-63).
inline constexpr scalar OMEGA_WF_DEFAULT_BETA1 = 0.075;
inline constexpr scalar OMEGA_WF_DEFAULT_CMU = 0.09;
inline constexpr scalar OMEGA_WF_DEFAULT_KAPPA = 0.41;

// Applies the omegaWallFunction kernel face-by-face.
//
// Matches Foam::omegaWallFunctionFvPatchScalarField::calculate() for the
// default blender (BINOMIAL, n=2) and updateCoeffs()'s write-back step
// (omega.C:485-527): each wall face contributes a blended omega to its
// owner cell and the BC face value is the same blended omega.
//
// Limitations vs upstream:
//  - Single-patch corner only: we assume each wall cell sees at most one
//    wall-function face, i.e. cornerWeights == 1. Multi-patch corner
//    averaging (omega.C:71-126) is not modelled — for typical meshes where
//    interior cells are bounded by ≤1 wall face this is exact; meshes with
//    re-entrant corners shared by two wall patches will diverge.
//  - G (production) accumulation is not applied here. Upstream's
//    calculate() adds (νₜ+ν)·|∇U_w|·C_µ^0.25·√k/(κ·y) into the turbulence
//    model's G field; NeoFOAM's kOmegaSST computes G internally from
//    nut·GbyNu0 and does not yet route it through boundary BCs. The
//    momentum/k near-wall production therefore won't match upstream
//    exactly even when omega values agree.
inline void setOmegaWallFunction(
    Field<scalar>& omega,
    const fvcc::VolumeField<scalar>& k,
    const fvcc::VolumeField<scalar>& nu,
    const fvcc::VolumeField<scalar>& nearWallDist,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    scalar beta1,
    scalar Cmu,
    scalar kappa
)
{
    const scalar Cmu25 = Kokkos::pow(Cmu, scalar(0.25));

    auto kInternal = k.internalVector().view();
    const auto nuBoundary = nu.boundaryData().value().view();
    const auto nearWallBoundary = nearWallDist.boundaryData().value().view();

    auto [refGrad, value, valueFraction, refValue, faceOwners] = views(
        omega.boundaryData().refGrad(),
        omega.boundaryData().value(),
        omega.boundaryData().valueFraction(),
        omega.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        omega.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = faceOwners[i];
            const scalar y = nearWallBoundary[i];
            const scalar nuw = nuBoundary[i];
            const scalar kw = Kokkos::max(kInternal[owner], scalar(0));

            // Viscous sublayer: 6ν / (β₁·y²)   (omega.C:218-224)
            const scalar wVis = 6.0 * nuw / (beta1 * y * y);

            // Log layer: √k / (C_µ^0.25·κ·y)   (omega.C:226-234)
            const scalar wLog = Kokkos::sqrt(kw) / (Cmu25 * kappa * y);

            // BINOMIAL blender, n = 2 (upstream default at omega.C:396) —
            // closed form: ω = √(ωᵥᵢₛ² + ωₗₒg²).
            const scalar wOmega = Kokkos::sqrt(wVis * wVis + wLog * wLog);

            // Set the wall face value only. Upstream also writes
            // omega_internal[wall_cell] via manipulateMatrix(setValues) — a
            // matrix-row pin we have no equivalent for. Without that pin a
            // direct stomp on omega.internalVector() only persists until the
            // next omega solve overwrites it; meanwhile correctNutInternal
            // reads the stomped value and collapses nut.internal at the wall
            // cell. Leaving the internal value alone keeps the fixedValue
            // semantics correct via the boundary face.
            value[i] = wOmega;
            refValue[i] = wOmega;
            valueFraction[i] = 1.0;
            refGrad[i] = 0.0;
        },
        "setOmegaWallFunction"
    );
}

} // namespace detail


// Mirrors Foam::omegaWallFunctionFvPatchScalarField
// (src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
//  omegaWallFunctions/omegaWallFunction/omegaWallFunctionFvPatchScalarField.{H,C}).
//
// Upstream is a fixedValueFvPatchField<scalar> whose updateCoeffs() blends a
// viscous-sublayer and a log-layer estimate of omega and also overwrites the
// adjacent wall cell's omega in the internal field. We replicate that here
// inside correctBoundaryCondition(ctx).
//
// Context fields required:
//   - "k"            VolumeField<scalar>  — to evaluate √k at the wall cell
//   - "nu"           VolumeField<scalar>  — laminar viscosity, boundary face values
//   - "nearWallDist" VolumeField<scalar>  — y for the wall cell, boundary face values
//
// Mirroring NutUSpaldingWallFunction (nutWallFunction.hpp:142-153), the no-arg
// correctBoundaryCondition() overload is a no-op: this BC must be driven from
// kOmegaSST::correct() (or equivalent) with a BoundaryContext that supplies
// the three fields above.
class OmegaWallFunction : public VolumeBoundaryFactory<scalar>::template Register<OmegaWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<OmegaWallFunction>;

public:

    OmegaWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true})
        , mesh_(mesh)
        , beta1_(
              dict.contains("beta1") ? dict.get<scalar>("beta1") : detail::OMEGA_WF_DEFAULT_BETA1
          )
        , Cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::OMEGA_WF_DEFAULT_CMU)
        , kappa_(
              dict.contains("kappa") ? dict.get<scalar>("kappa") : detail::OMEGA_WF_DEFAULT_KAPPA
          )
    {}

    void correctBoundaryCondition(Field<scalar>& /*domainVector*/) final {}

    void
    correctBoundaryCondition(Field<scalar>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        detail::setOmegaWallFunction(
            domainVector,
            ctx.scalarFieldPtr("k"),
            ctx.scalarFieldPtr("nu"),
            ctx.scalarFieldPtr("nearWallDist"),
            mesh_,
            this->range(),
            beta1_,
            Cmu_,
            kappa_
        );
    }

    static std::string name() { return "omegaWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Specific dissipation rate wall function. Blends viscous "
               "(6ν/(β₁y²)) and log-layer (√k/(C_µ^0.25·κ·y)) omega "
               "contributions with the BINOMIAL n=2 blender (upstream default) "
               "and sets the wall face value. Does NOT overwrite the wall cell "
               "internal value (upstream's setValues equivalent is absent). "
               "Single-patch-corner only; G is fed back through kOmegaSST.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<OmegaWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    scalar beta1_;
    scalar Cmu_;
    scalar kappa_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
