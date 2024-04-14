// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <memory>
#include <string>

namespace NeoFOAM
{

class surfaceInterpolationFactory
{
public:

    using factoryFunction = std::function<std::unique_ptr<surfaceInterpolationKernel>(const executor& exec, const unstructuredMesh& mesh)>;

    // surfaceInterpolationFactory();

    // static std::unique_ptr<surfaceInterpolationKernel> New(const std::string& className, const executor& exec, const unstructuredMesh& mesh);
    static std::unique_ptr<surfaceInterpolationKernel> New(const std::string& className, const executor& exec, const unstructuredMesh& mesh)
    {
        auto it = classMap.find(className);
        if (it != classMap.end())
        {
            return it->second(exec, mesh); // Call the creation lambda
        }
        std::cout << "Class not found" << std::endl;
        return nullptr;
    }

    static std::size_t number_of_instances()
    {
        return classMap.size();
    }


    template<typename T>
    static std::size_t registerClass(const std::string className, T classMapFunction)
    {
        auto const index = classMap.size();
        classMap.insert({className, classMapFunction});
        return index;
    }

private:

    static std::unordered_map<std::string, factoryFunction> classMap;
};

inline std::unordered_map<std::string, surfaceInterpolationFactory::factoryFunction> surfaceInterpolationFactory::classMap;


} // namespace NeoFOAM