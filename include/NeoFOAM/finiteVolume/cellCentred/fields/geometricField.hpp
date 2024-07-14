// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @brief Represents a geometric field for a cell-centered finite volume method.
 *
 * This class stores the field data for a cell-centered finite volume method. It contains
 * the internal field and boundary field, as well as the boundary conditions.
 *
 * @tparam ValueType The data type of the field.
 * @tparam BoundaryType The boundary type of the field.
 */
template<typename ValueType>
class GeometricFieldMixin
{
public:

    /**
     * @brief Constructor for GeometricField.
     *
     * @param exec The executor object.
     * @param mesh The unstructured mesh object.
     * @param boundaryConditions The vector of unique pointers to SurfaceBoundaryField objects
     * representing the boundary conditions.
     */
    GeometricFieldMixin(
        const Executor& exec, const UnstructuredMesh& mesh, const DomainField<ValueType>& field
    )
        : exec_(exec), mesh_(mesh), field_(field)
    {}

    /**
     * @brief Returns a const reference to the internal field.
     *
     * @return The const reference to the internal field.
     */
    const Field<ValueType>& internalField() const { return field_.internalField(); }

    /**
     * @brief Returns a reference to the internal field.
     *
     * @return The reference to the internal field.
     */
    Field<ValueType>& internalField() { return field_.internalField(); }

    /**
     * @brief Returns a const reference to the boundary field.
     *
     * @return The const reference to the boundary field.
     */
    const BoundaryFields<ValueType>& boundaryField() const { return field_.boundaryField(); }

    /**
     * @brief Returns a reference to the boundary field.
     *
     * @return The reference to the boundary field.
     */
    BoundaryFields<ValueType>& boundaryField() { return field_.boundaryField(); }

    /**
     * @brief Returns a const reference to the executor object.
     *
     * @return The const reference to the executor object.
     */
    const Executor& exec() const { return exec_; }

    /**
     * @brief Returns a const reference to the unstructured mesh object.
     *
     * @return The const reference to the unstructured mesh object.
     */
    const UnstructuredMesh& mesh() const { return mesh_; }

protected:

    Executor exec_;                // The executor object
    const UnstructuredMesh& mesh_; // The unstructured mesh object
    DomainField<ValueType> field_; // The domain field object
};

} // namespace NeoFOAM
