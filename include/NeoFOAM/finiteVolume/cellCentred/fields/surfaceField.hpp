// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @brief Represents a surface field for a cell-centered finite volume method.
 *
 * This class stores the surface field data for a cell-centered finite volume method. It contains
 * the internal field and boundary field, as well as the boundary conditions.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class SurfaceField
{
public:

    /**
     * @brief Constructor for SurfaceField.
     *
     * @param exec The executor object.
     * @param mesh The unstructured mesh object.
     * @param boundaryConditions The vector of unique pointers to SurfaceBoundaryField objects
     * representing the boundary conditions.
     */
    SurfaceField(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::vector<std::unique_ptr<SurfaceBoundaryField<ValueType>>>&& boundaryConditions
    )
        : exec_(exec), mesh_(mesh), field_(
                                        exec,
                                        mesh.nInternalFaces() + mesh.nBoundaryFaces(),
                                        mesh.nBoundaryFaces(),
                                        mesh.nBoundaries()
                                    ),
          boundaryConditions_(std::move(boundaryConditions))
    {}

    /**
     * @brief Copy constructor for SurfaceField.
     *
     * @param sField The SurfaceField object to be copied.
     */
    SurfaceField(const SurfaceField& sField)
        : exec_(sField.exec_), mesh_(sField.mesh_), field_(sField.field_),
          boundaryConditions_(sField.boundaryConditions_.size())
    {
        // Copy the boundary conditions
        // TODO add clone functionality to boundary conditions
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
            boundaryCondition->correctBoundaryConditions(
                field_.boundaryField(), field_.internalField()
            );
        }
    }

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

private:

    Executor exec_;                // The executor object
    const UnstructuredMesh& mesh_; // The unstructured mesh object
    DomainField<ValueType> field_; // The domain field object
    std::vector<std::unique_ptr<SurfaceBoundaryField<ValueType>>>
        boundaryConditions_; // The vector of boundary conditions
};

} // namespace NeoFOAM
