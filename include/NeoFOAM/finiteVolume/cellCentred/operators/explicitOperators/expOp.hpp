// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/tokenList.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/operators/explicitOperators/expOpDiv.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/explicitOperators/expOpDdt.hpp"

namespace dsl = NeoFOAM::DSL;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;


namespace NeoFOAM::finiteVolume::cellCentred::expOp
{

dsl::EqnTerm<NeoFOAM::scalar> ddt(fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return dsl::EqnTerm<NeoFOAM::scalar>(DdtScheme(phi));
}

dsl::EqnTerm<NeoFOAM::scalar>
ddt(fvcc::VolumeField<NeoFOAM::scalar>& rho, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    NF_ERROR_EXIT("Not implemented");
    return dsl::EqnTerm<NeoFOAM::scalar>(DdtScheme(phi));
}

dsl::EqnTerm<NeoFOAM::scalar> div(const fvcc::SurfaceField<NeoFOAM::scalar>& flux)
{
    NF_ERROR_EXIT("Not implemented");
    // return dsl::EqnTerm<NeoFOAM::scalar>(DivScheme(flux));
}

dsl::EqnTerm<NeoFOAM::scalar> div(fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    NF_ERROR_EXIT("Not implemented");
    return dsl::EqnTerm<NeoFOAM::scalar>(DdtScheme(phi)); // suppress warning
    // return dsl::EqnTerm<NeoFOAM::scalar>(DivScheme(flux));
}

dsl::EqnTerm<NeoFOAM::scalar>
div(const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux,
    fvcc::VolumeField<NeoFOAM::scalar>& phi,
    const NeoFOAM::Dictionary& dict)
{
    std::string schemeName = "div(" + faceFlux.name() + "," + phi.name() + ")";
    auto tokens = dict.subDict("divSchemes").get<NeoFOAM::TokenList>(schemeName);
    return dsl::EqnTerm<NeoFOAM::scalar>(DivScheme(faceFlux, phi, tokens));
}


} // namespace NeoFOAM
