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
 * @tparam T The value type of the field.
 */
template<ValueType T>
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
        const Executor& exec, const UnstructuredMesh& mesh, const DomainField<T>& field
    )
        : exec_(exec), mesh_(mesh), field_(field)
    {}

    /**
     * @brief Returns a const reference to the internal field.
     *
     * @return The const reference to the internal field.
     */
    const Field<T>& internalField() const { return field_.internalField(); }

    /**
     * @brief Returns a reference to the internal field.
     *
     * @return The reference to the internal field.
     */
    Field<T>& internalField() { return field_.internalField(); }

    /**
     * @brief Returns a const reference to the boundary field.
     *
     * @return The const reference to the boundary field.
     */
    const BoundaryFields<T>& boundaryField() const { return field_.boundaryField(); }

    /**
     * @brief Returns a reference to the boundary field.
     *
     * @return The reference to the boundary field.
     */
    BoundaryFields<T>& boundaryField() { return field_.boundaryField(); }

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
    DomainField<T> field_;         // The domain field object
};

} // namespace NeoFOAM
