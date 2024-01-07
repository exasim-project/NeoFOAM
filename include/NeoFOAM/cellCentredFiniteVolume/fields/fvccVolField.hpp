// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/blas/domainField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include <vector>

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
    template <typename T>
    class fvccVolField
    {

        public:

            /**
             * @brief Constructor for fvccVolField.
             * 
             * @param nCells The number of cells in the field.
             * @param nBoundaryFaces The number of boundary faces in the field.
             * @param nBoundaries The number of boundaries in the field.
             */
            fvccVolField(int nCells, int nBoundaryFaces, int nBoundaries, std::vector<std::unique_ptr<fvccBoundaryField<T> > > &&boundaryConditions)
                :  field_(nCells, nBoundaryFaces, nBoundaries),
                   boundaryConditions_(std::move(boundaryConditions))
            {
                
            };

            void correctBoundaryConditions()
            {
                for (auto &boundaryCondition : boundaryConditions_)
                {
                    boundaryCondition->correctBoundaryConditions(field_.boundaryField());
                }
            };

            const deviceField<T> &internalField() const
            {
                return field_.internalField();
            };
            
            deviceField<T> &internalField()
            {
                return field_.internalField();
            };


            const boundaryFields<T> &boundaryField() const
            {
                return field_.boundaryField();
            };

            boundaryFields<T> &boundaryField()
            {
                return field_.boundaryField();
            };

            std::vector<std::unique_ptr<fvccBoundaryField<T> > > &boundaryConditions()
            {
                return boundaryConditions_;
            };

        private:

            domainField<T> field_;
            std::vector<std::unique_ptr<fvccBoundaryField<T> > > boundaryConditions_;

    };

} // namespace NeoFOAM
