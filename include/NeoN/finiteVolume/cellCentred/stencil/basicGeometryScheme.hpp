// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

class BasicGeometryScheme : public GeometrySchemeFactory
{

public:

    BasicGeometryScheme(const UnstructuredMesh& mesh);

    void updateWeights(const Executor& exec, SurfaceField<scalar>& weights) override;

    void updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs) override;

    void updateNonOrthDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& nonOrthDeltaCoeffs)
        override;

    void updateNonOrthDeltaCoeffs(const Executor& exec, SurfaceField<Vector>& nonOrthDeltaCoeffs)
        override;


private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN
