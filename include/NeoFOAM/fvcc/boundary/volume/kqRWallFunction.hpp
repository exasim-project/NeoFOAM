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

// Mirrors OpenFOAM's kqRWallFunctionFvPatchField<scalar>
// (src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
//  kqRWallFunctions/kqRWallFunction/kqRWallFunctionFvPatchField.{H,C}).
//
// Upstream the class derives from zeroGradientFvPatchField<Type> and overrides
// nothing — its own docstring states "It is not a wall-function condition.".
// We mirror that here: correctBoundaryCondition() applies the same kernel as
// FixedGradient<scalar> with a zero gradient.
//
// Kept as a distinct dispatched class (not a string-level alias of
// fixedGradient) so OpenFOAM dictionaries authored with `type kqRWallFunction;`
// flow through the runtime-selection factory unchanged. This also leaves the
// door open for future wall functions (omegaWallFunction, nutkWallFunction)
// that *do* need a real wall-distance-based kernel — they can be registered
// alongside this class and consume BoundaryContext::scalarFieldPtr("nearWallDist")
// the way NutUSpaldingWallFunction already does.
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
