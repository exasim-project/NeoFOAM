// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <functional>
#include <type_traits>
#include <unordered_map>
#include <memory>
#include <string>

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/linear.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/upwind.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"

namespace NeoFOAM
{

SurfaceInterpolation surfaceInterpolationSelector(
    std::string interPolMethodName, const executor& exec, const UnstructuredMesh& mesh
);


class CompressionMethodFactory
{
public:

    // using TCreateMethod = std::unique_ptr<ICompressionMethod>(*)();
    using TCreateMethod = std::function<std::unique_ptr<SurfaceInterpolationKernel>(
        const executor& exec, const UnstructuredMesh& mesh
    )>;

public:

    CompressionMethodFactory() = delete;

    static bool register(const std::string name, TCreateMethod funcCreate);

    static surfaceInterpolation
    create(const std::string& name, const Executor& exec, const UnstructuredMesh& mesh);

    static int size() { return sMethods.size(); }

private:

    static std::unordered_map<std::string, TCreateMethod> sMethods;
};


} // namespace NeoFOAM
