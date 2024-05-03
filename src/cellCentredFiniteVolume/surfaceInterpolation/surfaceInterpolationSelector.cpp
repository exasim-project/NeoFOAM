// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationSelector.hpp"
#include "NeoFOAM/core/Error.hpp"

namespace NeoFOAM
{

std::unordered_map<std::string, CompressionMethodFactory::TCreateMethod> CompressionMethodFactory::s_methods;

bool CompressionMethodFactory::Register(const std::string name, CompressionMethodFactory::TCreateMethod funcCreate)
{
    if (auto it = s_methods.find(name); it == s_methods.end())
    { // C++17 init-if ^^
        s_methods[name] = funcCreate;
        return true;
    }
    return false;
}

surfaceInterpolation
CompressionMethodFactory::Create(const std::string& name, const executor& exec, const unstructuredMesh& mesh)
{
    if (auto it = s_methods.find(name); it != s_methods.end())
        return surfaceInterpolation(exec, mesh, it->second(exec, mesh));

    // return nullptr;
}
// surfaceInterpolationFactory::surfaceInterpolationFactory()
// {}
surfaceInterpolation surfaceInterpolationSelector(std::string interPolMethodName, const executor& exec, const unstructuredMesh& mesh)
{
    if (interPolMethodName == "upwind")
    {
        return surfaceInterpolation(exec, mesh, std::make_unique<upwind>(exec, mesh));
    }
    else if (interPolMethodName == "linear")
    {
        return surfaceInterpolation(exec, mesh, std::make_unique<linear>(exec, mesh));
    }
    else
    {
        error("not found").exit();
        return surfaceInterpolation(exec, mesh, std::make_unique<upwind>(exec, mesh)); // for the compiler
    }
}


} // namespace NeoFOAM
