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
 * @class GeometricFieldMixin
 * @brief This class represents a mixin for a geometric field.
 *
 * The GeometricFieldMixin class provides a set of common operations and accessors for a geometric
 * field. It is designed to be used as a mixin in other classes that require geometric field
 * functionality.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class GeometricFieldMixin
{
public:

    /**
     * @brief Constructor for GeometricFieldMixin.
     *
     * @param exec The executor object.
     * @param mesh The unstructured mesh object.
     * @param field The domain field object.
     */
    GeometricFieldMixin(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const DomainField<ValueType>& field
    )
        : exec_(exec), name_(name), mesh_(mesh), field_(field)
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

    /**
     * @brief Returns the name of the field.
     *
     * @return The name of the field.
     */
    std::string name() const { return name_; }

protected:

    Executor exec_;                // The executor object
    std::string name_;             // The name of the field
    const UnstructuredMesh& mesh_; // The unstructured mesh object
    DomainField<ValueType> field_; // The domain field object
};

} // namespace NeoFOAM
