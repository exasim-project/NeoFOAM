// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/fixedGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

/**
 * Zero-gradient boundary condition for turbulent kinetic energy k (and q, R) at walls.
 * Applies a zero gradient; equivalent to fixedGradient with gradient = 0.
 */
class KqRWallFunction : public VolumeBoundaryFactory<scalar>::template Register<KqRWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<KqRWallFunction>;

public:

    using Base::correctBoundaryCondition;

    KqRWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = false})
        , mesh_(mesh)
    {}

    void correctBoundaryCondition(Field<scalar>& domainVector) final
    {
        detail::setGradientValue(domainVector, mesh_, this->range(), scalar(0));
    }

    static std::string name() { return "kqRWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Renamed zero-gradient BC for turbulent kinetic energy k (and q, R) "
               "at high-Re walls. Matches OpenFOAM's kqRWallFunction, which is not "
               "actually a wall function — see upstream class docstring.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<KqRWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
