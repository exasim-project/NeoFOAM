// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/cellCentred.hpp"
#include "NeoFOAM/core/Error.hpp"

namespace NeoFOAM
{

template<>
std::unique_ptr<fvccBoundaryField<scalar>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{
    std::string type = patchDict.get<std::string>("type");
    if (type == "zeroGradient")
    {
        return std::make_unique<fvccScalarZeroGradientBoundaryField>(mesh, patchID);
    }
    else if (type == "fixedValue")
    {
        return std::make_unique<fvccScalarFixedValueBoundaryField>(
            mesh, patchID, patchDict.get<scalar>("value")
        );
    }
    else if (type == "empty")
    {
        return std::make_unique<fvccScalarEmptyBoundaryField>(mesh, patchID);
    }
    else if (type == "calculated")
    {
        return std::make_unique<fvccScalarCalculatedBoundaryField>(mesh, patchID);
    }
    else if (type == "extrapolatedCalculated")
    {
        // TODO create extrapolatedCalculated boundary condition
        return std::make_unique<fvccScalarCalculatedBoundaryField>(mesh, patchID);
    }
    else
    {
        std::cout << "keyword not found" << std::endl;
    }
    NF_ERROR_EXIT("Boundary condition not found");
    return std::make_unique<fvccScalarEmptyBoundaryField>(mesh, patchID);
}

template<>
std::unique_ptr<fvccBoundaryField<Vector>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{
    std::string type = patchDict.get<std::string>("type");
    if (type == "zeroGradient")
    {
        return std::make_unique<fvccVectorZeroGradientBoundaryField>(mesh, patchID);
    }
    else if (type == "fixedValue")
    {
        return std::make_unique<fvccVectorFixedValueBoundaryField>(
            mesh, patchID, patchDict.get<Vector>("value")
        );
    }
    else if (type == "empty")
    {
        return std::make_unique<fvccVectorEmptyBoundaryField>(mesh, patchID);
    }
    else if (type == "extrapolatedCalculated")
    {
        // TODO create extrapolatedCalculated boundary condition
        return std::make_unique<fvccVectorCalculatedBoundaryField>(mesh, patchID);
    }
    else if (type == "calculated")
    {
        return std::make_unique<fvccVectorCalculatedBoundaryField>(mesh, patchID);
    }
    NF_ERROR_EXIT("Boundary condition not found");
    return std::make_unique<fvccVectorEmptyBoundaryField>(mesh, patchID);
}

template<>
std::unique_ptr<fvccSurfaceBoundaryField<scalar>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{
    std::string type = patchDict.get<std::string>("type");
    if (type == "calculated")
    {
        return std::make_unique<fvccSurfaceScalarCalculatedBoundaryField>(mesh, patchID);
    }
    else if (type == "extrapolatedCalculated")
    {
        // TODO create extrapolatedCalculated boundary condition
        return std::make_unique<fvccSurfaceScalarCalculatedBoundaryField>(mesh, patchID);
    }
    else if (type == "empty")
    {
        return std::make_unique<fvccSurfaceScalarEmptyBoundaryField>(mesh, patchID);
    }
    NF_ERROR_EXIT("Boundary condition not found");
    return std::make_unique<fvccSurfaceScalarCalculatedBoundaryField>(mesh, patchID);
}


template<>
std::vector<std::unique_ptr<fvccSurfaceBoundaryField<scalar>>>
createCalculatedBCs(const UnstructuredMesh& mesh)
{
    const auto& bMesh = mesh.boundaryMesh();
    std::vector<std::unique_ptr<fvccSurfaceBoundaryField<scalar>>> bcs;

    for (int patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        Dictionary patchDict {};
        patchDict.insert(std::string("type"), std::string("calculated"));
        bcs.push_back(getSurfaceBC<scalar>(mesh, patchID, patchDict));
    }

    return bcs;
}


template<>
std::vector<std::unique_ptr<fvccSurfaceBoundaryField<Vector>>>
createCalculatedBCs(const UnstructuredMesh& mesh)
{
    std::vector<std::unique_ptr<fvccSurfaceBoundaryField<Vector>>> bcs;
    return bcs;
}


}; // namespace NeoFOAM
