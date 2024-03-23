// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructuredMesh/BoundaryMesh.hpp"

namespace NeoFOAM
{

BoundaryMesh::BoundaryMesh(
    const executor& exec,
    labelField faceCells,
    vectorField Cf,
    vectorField Cn,
    vectorField Sf,
    scalarField magSf,
    vectorField nf,
    vectorField delta,
    scalarField weights,
    scalarField deltaCoeffs,
    std::vector<localIdx> offset
)
    : exec_(exec),
      faceCells_(faceCells),
      Cf_(Cf),
      Cn_(Cn),
      Sf_(Sf),
      magSf_(magSf),
      nf_(nf),
      delta_(delta),
      weights_(weights),
      deltaCoeffs_(deltaCoeffs),
      offset_(offset) {

      };

// Accessor methods
const labelField& BoundaryMesh::faceCells() const
{
    return faceCells_;
}

std::span<const label> BoundaryMesh::faceCells(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return faceCells_.field().subspan(start, end - start);
}

const vectorField& BoundaryMesh::Cf() const
{
    return Cf_;
}

std::span<const Vector> BoundaryMesh::Cf(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return Cf_.field().subspan(start, end - start);
}

const vectorField& BoundaryMesh::Cn() const
{
    return Cn_;
}

std::span<const Vector> BoundaryMesh::Cn(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return Cn_.field().subspan(start, end - start);
}

const vectorField& BoundaryMesh::Sf() const
{
    return Sf_;
}

std::span<const Vector> BoundaryMesh::Sf(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return Sf_.field().subspan(start, end - start);
}

const scalarField& BoundaryMesh::magSf() const
{
    return magSf_;
}

std::span<const scalar> BoundaryMesh::magSf(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return magSf_.field().subspan(start, end - start);
}

const vectorField& BoundaryMesh::nf() const
{
    return nf_;
}

std::span<const Vector> BoundaryMesh::nf(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return nf_.field().subspan(start, end - start);
}

const vectorField& BoundaryMesh::delta() const
{
    return delta_;
}

std::span<const Vector> BoundaryMesh::delta(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return delta_.field().subspan(start, end - start);
}

const scalarField& BoundaryMesh::weights() const
{
    return weights_;
}

std::span<const scalar> BoundaryMesh::weights(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return weights_.field().subspan(start, end - start);
}

const scalarField& BoundaryMesh::deltaCoeffs() const
{
    return deltaCoeffs_;
}

std::span<const scalar> BoundaryMesh::deltaCoeffs(const int i) const
{
    label start = offset_[i];
    label end = offset_[i + 1];
    return deltaCoeffs_.field().subspan(start, end - start);
}

const std::vector<localIdx>& BoundaryMesh::offset() const
{
    return offset_;
}


} // namespace NeoFOAM