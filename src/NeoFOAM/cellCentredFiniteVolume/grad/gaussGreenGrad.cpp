// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/grad/gaussGreenGrad.hpp"
#include "NeoFOAM/blas/field.hpp"
#include <functional>

namespace NeoFOAM {
gaussGreenGrad::registerAlgorithm<std::function<void (NeoFOAM::vectorField &,const NeoFOAM::unstructuredMesh &,const scalarField &)> >("atomic", grad_atmoic);
gaussGreenGrad::registerAlgorithm<std::function<void (NeoFOAM::vectorField &,const NeoFOAM::unstructuredMesh &,const scalarField &)> >("not_atomic", grad_not_atmoic);
};