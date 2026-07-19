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

/// Default coefficients matching OpenFOAM's wallFunctionCoefficients defaults.
inline constexpr scalar EPSILON_WF_DEFAULT_CMU = 0.09;
inline constexpr scalar EPSILON_WF_DEFAULT_KAPPA = 0.41;

/**
 * @brief Apply the epsilonWallFunction kernel face-by-face.
 *
 * Mirrors Foam::epsilonWallFunctionFvPatchScalarField::calculate() for the
 * BINOMIAL n=2 blender:
 *   - ε_vis = 2·k·ν/y²            (viscous-sublayer)
 *   - ε_log = C_µ^0.75·k^1.5/(κ·y) (log-layer)
 *   - ε_w   = √(ε_vis² + ε_log²)  (blend)
 *
 * Only sets the wall face value; does NOT write to epsilon.internal[wall_cell].
 */
inline void setEpsilonWallFunction(
    Field<scalar>& epsilon,
    const fvcc::VolumeField<scalar>& k,
    const fvcc::VolumeField<scalar>& nu,
    const fvcc::VolumeField<scalar>& nearWallDist,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    scalar Cmu,
    scalar kappa
)
{
    const scalar Cmu75 = Kokkos::pow(Cmu, scalar(0.75));

    auto kInternal = k.internalVector().view();
    const auto nearWallBoundary = nearWallDist.boundaryData().value().view();
    (void)nu; // molecular viscosity only used by the (disabled) viscous-sublayer branch

    auto [refGrad, value, valueFraction, refValue, faceOwners] = views(
        epsilon.boundaryData().refGrad(),
        epsilon.boundaryData().value(),
        epsilon.boundaryData().valueFraction(),
        epsilon.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        epsilon.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = faceOwners[i];
            const scalar y = nearWallBoundary[i];
            const scalar kw = Kokkos::max(kInternal[owner], scalar(0));

            // STEPWISE blender (OpenFOAM v2406 default) with lowReCorrection off:
            // epsilon0 = epsilonLog = Cmu^0.75 k^1.5 / (kappa y). (The viscous
            // sublayer branch is only taken when lowReCorrection && yPlus < yPlusLam.)
            const scalar eLog = Cmu75 * Kokkos::pow(kw, scalar(1.5)) / (kappa * y);
            const scalar eOmega = eLog;

            value[i] = eOmega;
            refValue[i] = eOmega;
            valueFraction[i] = 1.0;
            refGrad[i] = 0.0;
        },
        "setEpsilonWallFunction"
    );
}

} // namespace detail

/**
 * @brief Turbulent dissipation rate wall function BC.
 *
 * Mirrors Foam::epsilonWallFunctionFvPatchScalarField: blends viscous and
 * log-layer ε estimates (BINOMIAL n=2) and sets the wall face value.
 *
 * Required BoundaryContext fields:
 *   - @c "k"            VolumeField<scalar> — turbulent kinetic energy
 *   - @c "nu"           VolumeField<scalar> — laminar viscosity, boundary values
 *   - @c "nearWallDist" VolumeField<scalar> — cell-to-wall distance y, boundary values
 *
 * Single-patch-corner only (cornerWeights == 1). G feedback is handled by
 * the parent KEpsilon model, not by this BC.
 */
class EpsilonWallFunction :
    public VolumeBoundaryFactory<scalar>::template Register<EpsilonWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<EpsilonWallFunction>;

public:

    EpsilonWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true})
        , mesh_(mesh)
        , Cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::EPSILON_WF_DEFAULT_CMU)
        , kappa_(
              dict.contains("kappa") ? dict.get<scalar>("kappa") : detail::EPSILON_WF_DEFAULT_KAPPA
          )
    {}

    void correctBoundaryCondition(Field<scalar>& /*domainVector*/) final {}

    void
    correctBoundaryCondition(Field<scalar>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        detail::setEpsilonWallFunction(
            domainVector,
            ctx.scalarFieldPtr("k"),
            ctx.scalarFieldPtr("nu"),
            ctx.scalarFieldPtr("nearWallDist"),
            mesh_,
            this->range(),
            Cmu_,
            kappa_
        );
    }

    static std::string name() { return "epsilonWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Dissipation rate wall function. Blends viscous "
               "(2νk/y²) and log-layer (C_µ^0.75·k^1.5/(κ·y)) ε contributions "
               "with the BINOMIAL n=2 blender (upstream default) and sets the "
               "wall face value. Does NOT overwrite the wall cell internal "
               "value. Single-patch-corner only; G is fed back through kEpsilon.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<EpsilonWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    scalar Cmu_;
    scalar kappa_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
