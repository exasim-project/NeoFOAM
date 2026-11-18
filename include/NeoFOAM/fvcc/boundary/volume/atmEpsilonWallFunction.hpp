// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/fvcc/boundary/volume/epsilonWallFunction.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail
{

/**
 * @brief Apply the atmEpsilonWallFunction kernel face-by-face.
 *
 * Mirrors Foam::atmEpsilonWallFunctionFvPatchScalarField::calculate() without
 * lowReCorrection (its default): the log-law value with the wall distance shifted by the
 * roughness length (PGVB:Eq. 7, RH:Eq. 8),
 *   ε_w = C_µ^0.75·k^1.5/(κ·(y + z0))
 *
 * Only sets the wall face value; the wall-adjacent cell pin and the G feedback are applied by
 * the KEpsilon model with the same y + z0.
 */
inline void setAtmEpsilonWallFunction(
    Field<scalar>& epsilon,
    const fvcc::VolumeField<scalar>& k,
    const fvcc::VolumeField<scalar>& nearWallDist,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    const Vector<scalar>& z0,
    scalar Cmu,
    scalar kappa
)
{
    const scalar Cmu75 = Kokkos::pow(Cmu, scalar(0.75));

    const auto kInternal = k.internalVector().view();
    const auto nearWallBoundary = nearWallDist.boundaryData().value().view();
    const auto z0V = z0.view();
    const localIdx start = range.first;

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
            const scalar yEff = nearWallBoundary[i] + z0V[i - start];
            const scalar kw = Kokkos::max(kInternal[owner], scalar(0));

            const scalar eLog = Cmu75 * Kokkos::pow(kw, scalar(1.5)) / (kappa * yEff);

            value[i] = eLog;
            refValue[i] = eLog;
            valueFraction[i] = 1.0;
            refGrad[i] = 0.0;
        },
        "setAtmEpsilonWallFunction"
    );
}

} // namespace detail

/**
 * @brief Rough-wall dissipation rate wall function for atmospheric flows.
 *
 * Mirrors Foam::atmEpsilonWallFunctionFvPatchScalarField: epsilonWallFunction with y replaced
 * by y + z0. The per-face roughness length is read from @c z0Values, which the reader evaluates
 * from OpenFOAM's z0 PatchFunction1. lowReCorrection is not supported (the reader rejects it).
 *
 * Required BoundaryContext fields:
 *   - @c "k"            VolumeField<scalar> — turbulent kinetic energy
 *   - @c "nearWallDist" VolumeField<scalar> — cell-to-wall distance y, boundary values
 */
class AtmEpsilonWallFunction :
    public VolumeBoundaryFactory<scalar>::template Register<AtmEpsilonWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<AtmEpsilonWallFunction>;

public:

    AtmEpsilonWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true})
        , mesh_(mesh)
        , z0_(mesh.exec(), dict.get<std::vector<scalar>>("z0Values"))
        , Cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::EPSILON_WF_DEFAULT_CMU)
        , kappa_(
              dict.contains("kappa") ? dict.get<scalar>("kappa") : detail::EPSILON_WF_DEFAULT_KAPPA
          )
    {
        if (toLocalIdx(z0_.size()) != this->patchSize())
        {
            NF_THROW(
                "'z0Values' holds " + std::to_string(z0_.size()) + " values but patch "
                + std::to_string(patchID) + " has " + std::to_string(this->patchSize()) + " faces"
            );
        }
    }

    void correctBoundaryCondition(Field<scalar>& /*domainVector*/) final {}

    void
    correctBoundaryCondition(Field<scalar>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        detail::setAtmEpsilonWallFunction(
            domainVector,
            ctx.scalarFieldPtr("k"),
            ctx.scalarFieldPtr("nearWallDist"),
            mesh_,
            this->range(),
            z0_,
            Cmu_,
            kappa_
        );
    }

    static std::string name() { return "atmEpsilonWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Rough-wall dissipation rate wall function for atmospheric boundary layers: "
               "C_µ^0.75·k^1.5/(κ·(y+z0)). The wall-cell pin and G feedback use the same y+z0 "
               "and are applied by kEpsilon.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<AtmEpsilonWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    Vector<scalar> z0_;
    scalar Cmu_;
    scalar kappa_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
