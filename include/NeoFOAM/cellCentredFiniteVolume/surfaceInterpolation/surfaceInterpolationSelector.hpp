// SPDX-License-Identifier: MIT
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
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"

namespace NeoFOAM
{

surfaceInterpolation surfaceInterpolationSelector(std::string interPolMethodName, const executor& exec, const unstructuredMesh& mesh);


class CompressionMethodFactory
{
public:

    // using TCreateMethod = std::unique_ptr<ICompressionMethod>(*)();
    using TCreateMethod = std::function<std::unique_ptr<surfaceInterpolationKernel>(const executor& exec, const unstructuredMesh& mesh)>;

public:

    CompressionMethodFactory() = delete;

    static bool Register(const std::string name, TCreateMethod funcCreate);

    static surfaceInterpolation Create(const std::string& name, const executor& exec, const unstructuredMesh& mesh);

    static int size() { return s_methods.size(); }

private:

    static std::unordered_map<std::string, TCreateMethod> s_methods;
};


} // namespace NeoFOAM
