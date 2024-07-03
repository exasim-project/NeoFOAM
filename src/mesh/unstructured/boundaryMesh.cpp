// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructured/boundaryMesh.hpp"

namespace NeoFOAM
{

BoundaryMesh::BoundaryMesh()
    : exec_(NeoFOAM::CPUExecutor()), faceCells_(exec_, 0), Cf_(exec_, 0), Cn_(exec_, 0),
      Sf_(exec_, 0), magSf_(exec_, 0), nf_(exec_, 0), delta_(exec_, 0), weights_(exec_, 0),
      deltaCoeffs_(exec_, 0), offset_() {};

BoundaryMesh::BoundaryMesh(
    const Executor& exec,
    labelField faceCells,
    vectorField cf,
    vectorField cn,
    vectorField sf,
    scalarField magSf,
    vectorField nf,
    vectorField delta,
    scalarField weights,
    scalarField deltaCoeffs,
    std::vector<localIdx> offset
)
    : exec_(exec), faceCells_(faceCells), Cf_(cf), Cn_(cn), Sf_(sf), magSf_(magSf), nf_(nf),
      delta_(delta), weights_(weights), deltaCoeffs_(deltaCoeffs), offset_(offset) {};

// Accessor methods
const labelField& BoundaryMesh::faceCells() const { return faceCells_; }

std::span<const label> BoundaryMesh::faceCells(const localIdx i) const
{
    const label& start = offset_[i];
    const label& end = offset_[i + 1];
    return faceCells_.span({start, end});
}

const vectorField& BoundaryMesh::cf() const { return Cf_; }

std::span<const Vector> BoundaryMesh::cf(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return Cf_.span({start, end});
}

const vectorField& BoundaryMesh::cn() const { return Cn_; }

std::span<const Vector> BoundaryMesh::cn(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return Cn_.span({start, end});
}

const vectorField& BoundaryMesh::sf() const { return Sf_; }

std::span<const Vector> BoundaryMesh::sf(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return Sf_.span({start, end});
}

const scalarField& BoundaryMesh::magSf() const { return magSf_; }

std::span<const scalar> BoundaryMesh::magSf(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return magSf_.span({start, end});
}

const vectorField& BoundaryMesh::nf() const { return nf_; }

std::span<const Vector> BoundaryMesh::nf(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return nf_.span({start, end});
}

const vectorField& BoundaryMesh::delta() const { return delta_; }

std::span<const Vector> BoundaryMesh::delta(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return delta_.span({start, end});
}

const scalarField& BoundaryMesh::weights() const { return weights_; }

std::span<const scalar> BoundaryMesh::weights(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return weights_.span({start, end});
}

const scalarField& BoundaryMesh::deltaCoeffs() const { return deltaCoeffs_; }

std::span<const scalar> BoundaryMesh::deltaCoeffs(const localIdx i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return deltaCoeffs_.span({start, end});
}

const std::vector<localIdx>& BoundaryMesh::offset() const { return offset_; }


} // namespace NeoFOAM
