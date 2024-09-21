// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// // forward declaration
template<typename GeoField>
class SolutionFields;

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

    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(
            exec,
            name,
            mesh,
            DomainField<ValueType>(exec, mesh.nCells(), mesh.nBoundaryFaces(), mesh.nBoundaries())
        ),
          boundaryConditions_(boundaryConditions)
    {}

    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions,
        SolutionFields<VolumeField<ValueType>>& solField,
        size_t solutionFieldKey
    )
        : GeometricFieldMixin<ValueType>(
            exec,
            name,
            mesh,
            DomainField<ValueType>(exec, mesh.nCells(), mesh.nBoundaryFaces(), mesh.nBoundaries())
        ),
        boundaryConditions_(boundaryConditions),
        solField_(solField),
        solutionFieldKey(solutionFieldKey)
    {}

    VolumeField(const VolumeField& other)
        : GeometricFieldMixin<ValueType>(other), boundaryConditions_(other.boundaryConditions_), solField_(other.solField_), solutionFieldKey(other.solutionFieldKey)
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

    /**
     * @brief Returns a const reference to the solution field object.
     *
     * @return The const reference to the solution field object.
    */
    const auto& solField() const { return solField_.value(); }

    /**
     * @brief Returns a reference to the solution field object.
     *
     * @return The reference to the solution field object.
    */
    auto& solField() { return solField_.value(); }

    bool hasSolField() const { return solField_.has_value(); }

    void setSolField(SolutionFields<VolumeField<ValueType>>& solField) { solField_ = solField; }

    std::optional<size_t> solutionFieldKey;

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<std::reference_wrapper<SolutionFields<VolumeField<ValueType>>>> solField_; // The solution field object
};

} // namespace NeoFOAM
