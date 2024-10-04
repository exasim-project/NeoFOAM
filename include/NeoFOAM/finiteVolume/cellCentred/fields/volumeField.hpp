// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @class VolumeField
 * @brief Represents a surface field in a finite volume method.
 *
 * The VolumeField class is a template class that represents a cell-centered field in a finite
 * volume method. It inherits from the GeometricFieldMixin class and provides methods for correcting
 * boundary conditions.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class VolumeField : public GeometricFieldMixin<ValueType>
{

public:

    /* @brief Constructor for a uninitialized VolumeField
     *
     * @param exec The executor
     * @param mesh The underlying mesh
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(
            exec,
            mesh,
            DomainField<ValueType>(exec, mesh.nCells(), mesh.nBoundaryFaces(), mesh.nBoundaries())
        ),
          boundaryConditions_(boundaryConditions)
    {}

    /* @brief Constructor for a VolumeField with a given internal field
     *
     * @param mesh The underlying mesh
     * @param internalField the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& internalField,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(exec, mesh, {exec, mesh, internalField}),
          boundaryConditions_(boundaryConditions)
    {}

    VolumeField(const VolumeField& other)
        : GeometricFieldMixin<ValueType>(other), boundaryConditions_(other.boundaryConditions_)
    {}

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
            boundaryCondition.correctBoundaryCondition(this->field_);
        }
    }

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
};

} // namespace NeoFOAM
