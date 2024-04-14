// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/primitives/label.hpp"

namespace NeoFOAM
{

class BoundaryMesh
{
public:

    BoundaryMesh(
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
    );


    const labelField& faceCells() const;

    std::span<const label> faceCells(const int i) const;

    const vectorField& Cf() const;

    std::span<const Vector> Cf(const int i) const;

    const vectorField& Cn() const;

    std::span<const Vector> Cn(const int i) const;

    const vectorField& Sf() const;

    std::span<const Vector> Sf(const int i) const;

    const scalarField& magSf() const;

    std::span<const scalar> magSf(const int i) const;

    const vectorField& nf() const;

    std::span<const Vector> nf(const int i) const;

    const vectorField& delta() const;

    std::span<const Vector> delta(const int i) const;

    const scalarField& weights() const;

    std::span<const scalar> weights(const int i) const;

    const scalarField& deltaCoeffs() const;

    std::span<const scalar> deltaCoeffs(const int i) const;

    const std::vector<localIdx>& offset() const;


private:

    const executor exec_;

    labelField faceCells_;
    vectorField Cf_;
    vectorField Cn_;
    vectorField Sf_;
    scalarField magSf_;
    vectorField nf_;
    vectorField delta_;
    scalarField weights_;
    scalarField deltaCoeffs_;

    std::vector<localIdx> offset_; ///< The view storing the offsets of each boundary.
};

} // namespace NeoFOAM