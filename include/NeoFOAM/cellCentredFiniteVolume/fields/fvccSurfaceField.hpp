// SPDX-License-Identifier: MPL-2.0
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
 * @brief A class representing a cell-centered finite volume field.
 *
 * This class is used to store and manipulate cell-centered finite volume fields.
 * It provides functionality for accessing and modifying the field values.
 *
 * @tparam T The type of the field values.
 */
template<typename T>
class fvccSurfaceField
{

public:

    /**
     * @brief Constructor for fvccSurfaceField.
     *
     * @param nCells The number of cells in the field.
     * @param nBoundaryFaces The number of boundary faces in the field.
     * @param nBoundaries The number of boundaries in the field.
     */
    fvccSurfaceField(
        const executor& exec,
        const unstructuredMesh& mesh,
        std::vector<std::unique_ptr<fvccSurfaceBoundaryField<T>>>&& boundaryConditions
    )
        : exec_(exec),
          mesh_(mesh),
          field_(exec, mesh.nInternalFaces() + mesh.nBoundaryFaces(), mesh.nBoundaryFaces(), mesh.nBoundaries()),
          boundaryConditions_(std::move(boundaryConditions)) {

          };

    fvccSurfaceField(const fvccSurfaceField& sField)
        : exec_(sField.exec_),
          mesh_(sField.mesh_),
          field_(sField.field_),
          boundaryConditions_(sField.boundaryConditions_.size()) {
              // for( const auto&: sField.boundaryConditions_)
              // {
              //     boundaryConditions_.push_back(std::make_unique<BoundaryConditionsType>(*sField.boundaryConditions_))
              // }
          };

    void correctBoundaryConditions()
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition->correctBoundaryConditions(field_.boundaryField(), field_.internalField());
        }
    };

    const Field<T>& internalField() const
    {
        return field_.internalField();
    };

    Field<T>& internalField()
    {
        return field_.internalField();
    };


    const boundaryFields<T>& boundaryField() const
    {
        return field_.boundaryField();
    };

    boundaryFields<T>& boundaryField()
    {
        return field_.boundaryField();
    };

    std::vector<std::unique_ptr<fvccSurfaceBoundaryField<T>>>& boundaryConditions()
    {
        return boundaryConditions_;
    };

    const executor& exec() const
    {
        return exec_;
    };

    const unstructuredMesh& mesh() const
    {
        return mesh_;
    };

private:

    executor exec_;
    const unstructuredMesh& mesh_;
    domainField<T> field_;
    std::vector<std::unique_ptr<fvccSurfaceBoundaryField<T>>> boundaryConditions_;
};

} // namespace NeoFOAM
