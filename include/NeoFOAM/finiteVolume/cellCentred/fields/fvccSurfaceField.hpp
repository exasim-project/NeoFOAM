// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccSurfaceBoundaryField.hpp"
#include <vector>
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{

/**
 * @brief Represents a surface field for a cell-centered finite volume method.
 *
 * This class stores the surface field data for a cell-centered finite volume method. It contains
 * the internal field and boundary field, as well as the boundary conditions.
 *
 * @tparam T The data type of the field.
 */
template<typename T>
class fvccSurfaceField
{
public:

    /**
     * @brief Constructor for fvccSurfaceField.
     *
     * @param exec The executor object.
     * @param mesh The unstructured mesh object.
     * @param boundaryConditions The vector of unique pointers to fvccSurfaceBoundaryField objects
     * representing the boundary conditions.
     */
    fvccSurfaceField(
        const executor& exec,
        const unstructuredMesh& mesh,
        std::vector<std::unique_ptr<fvccSurfaceBoundaryField<T>>>&& boundaryConditions
    )
        : exec_(exec), mesh_(mesh), field_(
                                        exec,
                                        mesh.nInternalFaces() + mesh.nBoundaryFaces(),
                                        mesh.nBoundaryFaces(),
                                        mesh.nBoundaries()
                                    ),
          boundaryConditions_(std::move(boundaryConditions)) {

          };

    /**
     * @brief Copy constructor for fvccSurfaceField.
     *
     * @param sField The fvccSurfaceField object to be copied.
     */
    fvccSurfaceField(const fvccSurfaceField& sField)
        : exec_(sField.exec_), mesh_(sField.mesh_), field_(sField.field_),
          boundaryConditions_(sField.boundaryConditions_.size()) {
              // Copy the boundary conditions
              // TODO add clone functionality to boundary conditions
          };

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
            boundaryCondition->correctBoundaryConditions(
                field_.boundaryField(), field_.internalField()
            );
        }
    };

    /**
     * @brief Returns a const reference to the internal field.
     *
     * @return The const reference to the internal field.
     */
    const Field<T>& internalField() const { return field_.internalField(); };

    /**
     * @brief Returns a reference to the internal field.
     *
     * @return The reference to the internal field.
     */
    Field<T>& internalField() { return field_.internalField(); };

    /**
     * @brief Returns a const reference to the boundary field.
     *
     * @return The const reference to the boundary field.
     */
    const boundaryFields<T>& boundaryField() const { return field_.boundaryField(); };

    /**
     * @brief Returns a reference to the boundary field.
     *
     * @return The reference to the boundary field.
     */
    boundaryFields<T>& boundaryField() { return field_.boundaryField(); };

    /**
     * @brief Returns a const reference to the executor object.
     *
     * @return The const reference to the executor object.
     */
    const executor& exec() const { return exec_; };

    /**
     * @brief Returns a const reference to the unstructured mesh object.
     *
     * @return The const reference to the unstructured mesh object.
     */
    const unstructuredMesh& mesh() const { return mesh_; };

private:

    executor exec_;                // The executor object
    const unstructuredMesh& mesh_; // The unstructured mesh object
    domainField<T> field_;         // The domain field object
    std::vector<std::unique_ptr<fvccSurfaceBoundaryField<T>>>
        boundaryConditions_; // The vector of boundary conditions
};

} // namespace NeoFOAM
