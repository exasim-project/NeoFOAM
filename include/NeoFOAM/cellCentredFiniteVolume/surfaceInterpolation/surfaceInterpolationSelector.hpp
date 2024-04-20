// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <memory>
#include <string>
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/linear.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/upwind.hpp"

namespace NeoFOAM
{

surfaceInterpolation surfaceInterpolationSelector(std::string interPolMethodName,const executor& exec, const unstructuredMesh& mesh);


} // namespace NeoFOAM