// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @class SurfaceField
 * @brief Represents a surface field in a finite volume method.
 *
 * The SurfaceField class is a template class that represents a face-centered field in a finite
 * volume method. It inherits from the GeometricFieldMixin class and provides methods for correcting
 * boundary conditions.
 *
 * @tparam T The value type of the field.
 */
template<ValueType T>
class SurfaceField : public GeometricFieldMixin<T>
{

public:

    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::vector<std::unique_ptr<SurfaceBoundary<T>>>&& boundaryConditions
    )
        : GeometricFieldMixin<T>(
            exec,
            mesh,
            DomainField<T>(
                exec,
                mesh.nInternalFaces() + mesh.nBoundaryFaces(),
                mesh.nBoundaryFaces(),
                mesh.nBoundaries()
            )
        ),
          boundaryConditions_(std::move(boundaryConditions))
    {}

    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const DomainField<T>& field,
        std::vector<std::unique_ptr<SurfaceBoundary<T>>>&& boundaryConditions
    )
        : GeometricFieldMixin<T>(exec, mesh, field),
          boundaryConditions_(std::move(boundaryConditions))
    {}

    SurfaceField(const SurfaceField& other)
        : GeometricFieldMixin<T>(other), boundaryConditions_(other.boundaryConditions_.size())
    {
        // for (size_t i = 0; i < other.boundaryConditions_.size(); ++i)
        // {
        //     boundaryConditions_[i] =
        //     std::make_unique<SurfaceBoundary<T>>(*other.boundaryConditions_[i]);
        // }
    }

    /**
     * @brief Corrects the boundary conditions of the surface field.
     *
     * This function applies the correctBoundaryConditions() method to each boundary condition in
     * the field.
     */
    void correctBoundaryConditions()
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition->correctBoundaryCondition(this->field_);
        }
    }

private:

    std::vector<std::unique_ptr<SurfaceBoundary<T>>>
        boundaryConditions_; // The vector of boundary conditions
};


} // namespace NeoFOAM
